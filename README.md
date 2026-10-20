


📄 Resume Screening Agent
A Streamlit web app that screens resumes against a job description. It scores and ranks candidates, applies a CGPA filter, and generates interview questions and email drafts for the shortlist. It works with the Gemini API, and also offline with keyword matching when no API key is available.
Features
Upload multiple resumes (PDF or TXT) and paste a job description
AI scoring with Google Gemini (score out of 100, matched and missing skills, summary)
Keyword-matching fallback when there is no API key or the AI call fails
Adjustable shortlist cutoff and minimum CGPA filter, with live Shortlist / Maybe / Reject decisions
Extracts candidate name, email and CGPA from the resume
Interview questions for shortlisted candidates (3 technical, 1 skill gap, 1 behavioral)
Email drafts (interview invite or polite rejection), editable in the app
Download results as CSV and email drafts as TXT
Tech Stack
Python, Streamlit, Google Gemini API (google-genai), pandas, pypdf
Project Structure
resume_agent/
├── app.py
├── requirements.txt



    It saves recruiters from reading hundreds of resumes by hand:
Saves time: one click scores and ranks all resumes against the job description.
Consistent: every candidate is judged by the same rules (score cutoff and CGPA filter), not by mood or fatigue.
Goes beyond ranking: it also produces interview questions and email drafts for shortlisted candidates.
Always works: if the AI is unavailable, keyword matching keeps it running.
For an interview, you can say: "I built it to cut down the time spent on initial resume screening, so recruiters can filter quickly while the final decision stays with a human."
