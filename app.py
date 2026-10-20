import json
import os
import re

import pandas as pd
import streamlit as st
from google import genai
from pypdf import PdfReader

st.set_page_config(page_title="Resume Screening Agent", page_icon="📄", layout="wide")
st.title("📄 Resume Screening Agent")
st.caption("Paste a job description, upload resumes, and get a ranked shortlist with "
           "interview questions and email drafts. Works without an API key too.")

# Models are tried one by one until one works
MODELS = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.0-flash"]

STOP_WORDS = {"the", "and", "for", "with", "you", "our", "are", "will", "have", "that",
              "this", "from", "your", "who", "has", "can", "all", "any", "job", "role"}

SCREEN_PROMPT = """You are an HR recruiter. Compare the resume with the job description.

JOB DESCRIPTION:
{jd}

MUST-HAVE SKILLS: {must}

RESUME:
{resume}

Return ONLY JSON:
{{"name": "candidate name", "score": integer from 0 to 100,
"matched_skills": ["..."], "missing_skills": ["..."],
"cgpa": "CGPA out of 10 or null", "percentage": "degree percentage or null",
"email": "candidate email or null", "summary": "2 short lines on why this score"}}"""

QUESTION_PROMPT = """You are a technical interviewer. Write 5 interview questions.

JOB DESCRIPTION:
{jd}

RESUME:
{resume}

Skills missing from the resume: {missing}

Rules: 3 technical questions from the candidate's real projects, 1 about a skill gap,
1 behavioral. Each under 30 words.
Return ONLY JSON: {{"questions": ["...", "...", "...", "...", "..."]}}"""

EMAIL_PROMPT = """Write a professional {kind} email from a recruiter to a candidate.

Candidate: {name}
Company: {company}
Role: {role}
Recruiter: {hr}
Candidate's matched skills: {skills}

Rules: under 120 words, polite, do not invent dates, times or salary.
Return ONLY JSON: {{"subject": "...", "body": "..."}}"""


# ---------------- small helpers ----------------
def read_file(f):
    """Get text from a PDF or TXT upload. Returns '' if it fails."""
    try:
        if f.name.lower().endswith(".txt"):
            return f.read().decode("utf-8", errors="ignore")
        return "\n".join(page.extract_text() or "" for page in PdfReader(f).pages)
    except Exception:
        return ""


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def as_list(x):
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str) and x.strip():
        return [s.strip() for s in x.split(",")]
    return []


def fix_cgpa(cgpa, pct=None):
    """Return CGPA out of 10. A percentage is converted with pct / 9.5."""
    c, p = to_float(cgpa), to_float(pct)
    if c is not None and c > 10:  # percentage written as CGPA
        c, p = None, c
    if c is not None and 0 < c <= 10:
        return round(c, 1)
    if p is not None and 0 < p <= 100:
        return round(p / 9.5, 1)
    return None


def find_email(text):
    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text or "")
    return m.group(0) if m else None


def find_cgpa(text):
    m = re.search(r"cgpa[^0-9]{0,12}(\d+(?:\.\d+)?)", (text or "").lower())
    return m.group(1) if m else None


def ask_gemini(client, prompt):
    """Ask Gemini for JSON. Tries each model, raises if all fail."""
    for model in MODELS:
        try:
            r = client.models.generate_content(
                model=model, contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0.3})
            text = re.sub(r"```(?:json)?", "", r.text or "").strip()
            return json.loads(text)
        except Exception:
            continue
    raise RuntimeError("Gemini failed")


# ---------------- screening ----------------
def keyword_screen(resume, jd, must):
    """Offline fallback: score = % of required skills found in the resume."""
    low = resume.lower()
    skills = [s.strip().lower() for s in must.split(",") if s.strip()]
    if not skills:  # no must-have skills given, take keywords from the JD
        words = re.findall(r"[a-zA-Z+#.]{3,}", jd.lower())
        skills = list(dict.fromkeys(w for w in words if w not in STOP_WORDS))[:25]
    matched = [s for s in skills if s in low]
    missing = [s for s in skills if s not in low]
    score = round(100 * len(matched) / len(skills)) if skills else 0
    first_line = next((l.strip() for l in resume.splitlines() if l.strip()), "Unknown")
    return {"name": first_line[:40], "score": score,
            "matched_skills": matched, "missing_skills": missing,
            "summary": f"Keyword match: {len(matched)}/{len(skills)} skills found."}


def screen(client, text, jd, must):
    """AI score if possible, otherwise keyword score."""
    if client:
        try:
            d = ask_gemini(client, SCREEN_PROMPT.format(
                jd=jd[:6000], must=must or "not specified", resume=text[:12000]))
            d["mode"] = "AI"
            return d
        except Exception:
            pass
    d = keyword_screen(text, jd, must)
    d["mode"] = "Keyword"
    return d


def decide(row, cutoff, min_cgpa):
    """Shortlist / Maybe / Reject from the score cutoff and the CGPA filter."""
    score = row["score"]
    decision = "Shortlist" if score >= cutoff else "Maybe" if score >= cutoff - 20 else "Reject"
    note = ""
    if min_cgpa > 0:
        if row["cgpa"] is None:
            note = "CGPA not found"
        elif row["cgpa"] < min_cgpa:
            decision, note = "Reject", f"CGPA {row['cgpa']} below {min_cgpa}"
    return decision, note


# ---------------- interview questions ----------------
def make_questions(client, row, resume_text, jd):
    if client:
        try:
            d = ask_gemini(client, QUESTION_PROMPT.format(
                jd=jd[:4000], resume=resume_text[:9000],
                missing=", ".join(row["missing_skills"][:5]) or "none"))
            qs = as_list(d.get("questions"))
            if qs:
                return qs[:5], "AI"
        except Exception:
            pass
    qs = [f"Explain a project where you used {s}. What would you improve?"
          for s in row["matched_skills"][:3]]
    if row["missing_skills"]:
        qs.append(f"This role needs {row['missing_skills'][0]}. How would you learn it quickly?")
    qs.append("Walk me through your most challenging project and your exact contribution.")
    qs.append("Tell me about a time you handled a disagreement in a team.")
    return qs[:5], "Template"


# ---------------- email drafts ----------------
def make_email(client, row, kind, company, role, hr):
    """kind is 'invite' or 'reject'. Returns (subject, body)."""
    if client:
        try:
            d = ask_gemini(client, EMAIL_PROMPT.format(
                kind="interview invitation" if kind == "invite" else "polite rejection",
                name=row["name"], company=company, role=role, hr=hr,
                skills=", ".join(row["matched_skills"][:4]) or "not specified"))
            if d.get("subject") and d.get("body"):
                return str(d["subject"]).strip(), str(d["body"]).strip()
        except Exception:
            pass
    name = row["name"]
    if kind == "invite":
        skills = ", ".join(row["matched_skills"][:3])
        line = f" Your background in {skills} stood out to us." if skills else ""
        return (f"Interview Invitation - {role} at {company}",
                f"Dear {name},\n\nThank you for applying for the {role} position at {company}.{line} "
                "We are pleased to inform you that you have been shortlisted for the next round.\n\n"
                "Please reply with your availability for an interview over the next few days, "
                f"and we will confirm the schedule.\n\nBest regards,\n{hr}\n{company}")
    return (f"Your application for {role} at {company}",
            f"Dear {name},\n\nThank you for your interest in the {role} position at {company}. "
            "After careful review, we will not be moving forward with your application at this time.\n\n"
            "We encourage you to apply for future openings that match your skills. "
            f"We wish you the very best.\n\nBest regards,\n{hr}\n{company}")


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key (optional)", type="password",
                            value=os.environ.get("GEMINI_API_KEY", ""))
    offline = st.checkbox("Offline mode (keyword matching only)")

    st.header("🎯 Screening rules")
    cutoff = st.slider("Shortlist cutoff (score)", 0, 100, 65)
    min_cgpa = st.slider("Minimum CGPA (0 = no filter)", 0.0, 10.0, 0.0, 0.1)

    with st.expander("🏢 Company details (for emails)"):
        company = st.text_input("Company name", value="Company Name")
        role = st.text_input("Role title", value="Graduate Engineer Trainee")
        hr_name = st.text_input("Recruiter name", value="HR Team")

client = None
if api_key.strip() and not offline:
    try:
        client = genai.Client(api_key=api_key.strip())
    except Exception:
        client = None

# ---------------- inputs ----------------
col1, col2 = st.columns(2)
with col1:
    jd = st.text_area("Job Description / Company Requirements", height=250,
                      placeholder="Role, must-have skills, experience, education...")
    must = st.text_input("Must-have skills (comma separated)",
                         placeholder="Python, SQL, Machine Learning")
with col2:
    files = st.file_uploader("Resumes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)

# ---------------- run screening ----------------
if st.button("🚀 Screen Resumes", type="primary"):
    if not jd.strip():
        st.info("Please enter a job description.")
    elif not files:
        st.info("Please upload at least one resume.")
    else:
        if client is None:
            st.info("AI is off, so keyword matching is being used.")
        rows, texts, notes = [], {}, []
        bar = st.progress(0.0)
        for i, f in enumerate(files):
            text = read_file(f)
            if len(text.strip()) < 30:
                notes.append(f"⚠️ {f.name}: could not read text (scanned or empty file). Skipped.")
            else:
                d = screen(client, text, jd, must)
                try:
                    score = max(0, min(100, int(float(d.get("score", 0)))))
                except Exception:
                    score = 0
                rows.append({
                    "id": i,
                    "file": f.name,
                    "name": str(d.get("name") or f.name.rsplit(".", 1)[0]),
                    "score": score,
                    "cgpa": fix_cgpa(d.get("cgpa"), d.get("percentage")) or fix_cgpa(find_cgpa(text)),
                    "email": d.get("email") or find_email(text),
                    "summary": str(d.get("summary") or ""),
                    "matched_skills": as_list(d.get("matched_skills")),
                    "missing_skills": as_list(d.get("missing_skills")),
                    "mode": d["mode"],
                })
                texts[i] = text[:12000]
            bar.progress((i + 1) / len(files))

        # save results and clear old questions / drafts
        for key in list(st.session_state.keys()):
            if key.startswith(("subject_", "body_")):
                del st.session_state[key]
        st.session_state.update(rows=rows, texts=texts, notes=notes, jd=jd,
                                questions={}, drafts=[])

# ---------------- show results ----------------
rows = st.session_state.get("rows")
if rows is not None:
    for n in st.session_state.get("notes", []):
        st.warning(n)
    if not rows:
        st.info("No resumes could be read. Please upload text-based PDFs.")
        st.stop()

    # decisions change live when the sliders move
    data = []
    for r in rows:
        decision, note = decide(r, cutoff, min_cgpa)
        data.append({**r, "decision": decision, "note": note})
    df = pd.DataFrame(data).sort_values("score", ascending=False).reset_index(drop=True)
    df.index += 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Screened", len(df))
    c2.metric("Shortlisted", int((df["decision"] == "Shortlist").sum()))
    c3.metric("Maybe", int((df["decision"] == "Maybe").sum()))
    c4.metric("Rejected", int((df["decision"] == "Reject").sum()))

    st.subheader("🏆 Ranking")
    table = df[["name", "score", "decision", "cgpa", "note", "summary", "file"]]
    st.dataframe(table)
    st.download_button("⬇️ Download results (CSV)", table.to_csv().encode("utf-8"),
                       "screening_results.csv", "text/csv")

    st.subheader("🔍 Details")
    for r in df.to_dict("records"):
        with st.expander(f"{r['name']} — {r['score']}/100 ({r['decision']})"):
            st.write(r["summary"])
            st.write("✅ Matched:", ", ".join(r["matched_skills"]) or "-")
            st.write("❌ Missing:", ", ".join(r["missing_skills"]) or "-")
            if r["note"]:
                st.write("📌 Note:", r["note"])

    # ---- interview questions ----
    st.subheader("🎤 Interview questions")
    shortlisted = [r for r in data if r["decision"] == "Shortlist"]
    if st.button("Generate questions for shortlisted candidates"):
        if not shortlisted:
            st.info("No shortlisted candidates. Try lowering the cutoff.")
        else:
            with st.spinner("Writing questions..."):
                for r in shortlisted:
                    st.session_state["questions"][r["id"]] = make_questions(
                        client, r, st.session_state["texts"].get(r["id"], ""), st.session_state["jd"])
    for r in shortlisted:
        saved = st.session_state["questions"].get(r["id"])
        if saved:
            with st.expander(f"Questions for {r['name']} ({saved[1]})"):
                for n, q in enumerate(saved[0], 1):
                    st.write(f"{n}. {q}")

    # ---- email drafts ----
    st.subheader("✉️ Email drafts")
    st.caption("Shortlisted candidates get an interview invite, rejected ones get a polite regret email. "
               "You can edit them here and download all drafts as a TXT file.")
    if st.button("Generate email drafts"):
        targets = [r for r in data if r["decision"] in ("Shortlist", "Reject")]
        if not targets:
            st.info("No shortlisted or rejected candidates.")
        else:
            for key in list(st.session_state.keys()):
                if key.startswith(("subject_", "body_")):
                    del st.session_state[key]
            drafts = []
            with st.spinner("Drafting emails..."):
                for r in targets:
                    kind = "invite" if r["decision"] == "Shortlist" else "reject"
                    subject, body = make_email(client, r, kind, company, role, hr_name)
                    drafts.append({"id": r["id"], "name": r["name"], "to": r["email"],
                                   "kind": kind, "subject": subject, "body": body})
            st.session_state["drafts"] = drafts

    drafts = st.session_state.get("drafts", [])
    for d in drafts:
        label = "Invite" if d["kind"] == "invite" else "Rejection"
        with st.expander(f"{label} → {d['name']} ({d['to'] or 'no email found'})"):
            st.text_input("Subject", d["subject"], key=f"subject_{d['id']}")
            st.text_area("Body", d["body"], height=220, key=f"body_{d['id']}")

    if drafts:
        parts = []
        for d in drafts:
            subject = st.session_state[f"subject_{d['id']}"]
            body = st.session_state[f"body_{d['id']}"]
            parts.append(f"To: {d['to']}\nSubject: {subject}\n\n{body}\n" + "-" * 40)
        text_file = "\n\n".join(parts)
        st.download_button("⬇️ Download email drafts (TXT)", text_file.encode("utf-8"),
                           "email_drafts.txt", "text/plain")


    #cd "C:\Users\pc\Favorites\Downloads\New folder"
    #conda activate screener
    #streamlit run app.py --theme.base dark
