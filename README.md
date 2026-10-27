CivicFix Campus: Phase Review Report
Student: Sam Paul Kirubakaran R Institution: Rathinam Technical Campus, Coimbatore Project: CivicFix Campus, an AI-assisted, multi-campus problem reporting and resolution platform Repository: https://github.com/paulkirubakaran2008-sam/civicfix-campus Tagline: Report. Prioritize. Resolve. Improve.

1. Problem and Solution
Campus maintenance issues (potholes, overflowing bins, broken streetlights, leaks) are usually reported through informal channels. Reports get lost, duplicates pile up, and staff have no way to tell what is urgent. CivicFix gives students one place to report an issue with a photo. Each report is analysed by AI, scored by an explainable priority engine, and shown to campus staff in a ranked queue.

2. Architecture
React + Vite frontend --(HTTPS, JWT bearer token)--> FastAPI backend
                                                        |
                    Gemini vision API (if key set) --falls back to--> mock analysis
                                                        |
                              MongoDB (Motor), or in-memory fallback when unset
Images: Cloudinary (if configured), otherwise local uploads/ folder
Layer	Technology
Frontend	React, Vite, Leaflet (map), Recharts (analytics)
Backend	Python, FastAPI, Uvicorn
Database	MongoDB via Motor, with in-memory fallback
AI	Google Gemini vision API, with deterministic fallback
Auth	JWT, student and admin roles
Tests	pytest
3. Completed Modules
Module	Files	What it does
Authentication and roles	backend/app/auth.py, routes/auth.py	JWT login; admin-only routes return 403 to students
Report submission	routes/reports.py	Category, location, description and photo in one form; rejects bad file types (415) and files over 5 MB (413)
AI analysis (Gemini)	services/gemini_analysis.py	Sends the uploaded photo to Gemini and parses a structured JSON assessment; the API key is backend-only
AI fallback	services/analysis.py	Deterministic provider used when Gemini is unset, errors, times out or hits quota, so submission never fails
Priority engine	services/priority_engine.py	0-100 score: severity 30%, safety risk 25%, people affected 20%, supporting reports 15%, location 10%, with a visible breakdown
Duplicate detection and support	services/duplicates.py, /reports/duplicate-check, /reports/{id}/support	Surfaces similar reports; supporting one raises its priority
Admin workflow	routes/admin.py	Ranked queue, assignment, status and resolution notes
Map and analytics	routes/insights.py	Priority-coloured map markers; breakdowns by category, location, status, trend and average resolution time
Recurring-problem prediction	services/predictions.py	Frequency-based model once six or more reports exist; clearly labelled demo signals below that
Multi-campus isolation	campuses.py, repository.py	Every query is scoped by campus_id at the data layer, not in the UI
Image storage	services/image_storage.py	Cloudinary upload when configured, local disk otherwise
Deployment config	backend/render.yaml, frontend/vercel.json, .env.example files	Environment-driven settings; CORS restricted to one configured origin
4. Evidence
Each result below can be reproduced.

Backend tests: 8 of 8 pass (cd backend && pytest). They cover login and rejection of wrong passwords, report submission, urgency-aware analysis, the duplicate, support and admin workflow, map and analytics feeds, the prediction fallback, rejection of invalid files, and campus isolation.

Frontend production build succeeds (cd frontend && npm run build).

Health check: GET /health returns {"status":"ok", ...}.

Secret scan: a scan of all tracked files for API keys, connection strings and private keys came back clean. .env, .venv, node_modules and uploads/ are gitignored.

Multi-campus isolation test
test_campus_isolation creates a report as a Main Campus student, then checks:

a North Gate student does not see it in their list
a North Gate admin does not see it in the admin queue
a North Gate admin gets HTTP 404 when assigning it by direct ID
the Main Campus admin can assign it
The 404 on a direct ID shows the restriction is enforced on the server, not just hidden in the interface.

Demo accounts
Campus	Role	Email	Password
Main Campus	Student	student@civicfix.edu	Student@123
Main Campus	Admin	admin@civicfix.edu	Admin@123
North Gate Campus	Student	student.north@civicfix.edu	Student@123
North Gate Campus	Admin	admin.north@civicfix.edu	Admin@123
5. Engineering Decisions
Fallback instead of failure. If the AI provider is down, the report is still saved using the fallback. Every report records which provider analysed it (gemini-vision-v1 or mock-explainable-v1), so a fallback result is never passed off as real image analysis.
One scoring module. Both providers feed priority_engine.py, so the weighting is defined once.
Secrets stay on the server. The Gemini key is never sent to the browser or returned in an API response. Gemini error bodies are not echoed to clients.
Bug found and fixed. The in-memory store was not reset between app startups, so test data leaked between tests and broke an analytics test. It now starts empty each run.
6. Pending Work and Known Limitations
Item	Status
Public deployment on Vercel, Render and MongoDB Atlas	 done. Config files exist,  accounts are set up, so there is URL
Live Gemini test	Not done. The integration is written against current documentation but has not yet called the real API. It must be tested with a real key and photo; success means a report returns ai_provider: gemini-vision-v1
Cloudinary upload test	Not done, for the same reason
Rate limiting on AI endpoints	Not implemented
Page metadata (SEO)	Not implemented
Testing on real phones and tablets	Not done
Demo accounts use plaintext passwords	Acceptable for four fake accounts; hash_password and verify_password helpers are ready for a migration
POST /reports/{id}/analyze	Uses the fallback provider only
CORS	Allows one origin at a time
Note: gemini-2.5-flash has a scheduled retirement on 16 October 2026, so GEMINI_MODEL must be checked and updated before the live test.

7. Next Steps
Create free-tier accounts (MongoDB Atlas, Google AI Studio, Render, Vercel) and deploy
Set FRONTEND_ORIGIN on the backend to the deployed frontend URL
Run the live Gemini test with a real pothole photo
Run the 16-point production checklist in DEPLOYMENT.md, including cross-campus isolation on the live site
Add rate limiting and page metadata
Move demo accounts to hashed, persisted users
8. Summary
The application works end to end locally and is covered by 8 passing tests. Its most security-sensitive part, multi-campus isolation, is enforced at the data layer and has its own test. Against the full deployment brief I estimate it is roughly 45-50% complete. The main gaps are the live deployment and the live Gemini verification, and I have not claimed either as done.


