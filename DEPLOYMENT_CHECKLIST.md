# Quiz Helper - Deployment Checklist

## ✅ Pre-Deployment Verification

### Files and Structure
- [x] `main.py` - Python backend with FastAPI
- [x] `templates/index.html` - Frontend with HTML + JavaScript + CSS
- [x] `static/style.css` - Styling rules
- [x] `requirements.txt` - Python dependencies
- [x] `Procfile` - Render deployment config
- [x] `render.yaml` - Render config file (optional)
- [x] `.env` - Supabase credentials (kept locally)
- [x] `supabase/migrations/` - Database migrations

### Backend Verification
- [x] Python syntax valid
- [x] All imports present (httpx, pdfplumber, beautifulsoup4, fastapi, uvicorn)
- [x] Memory optimization: no SESSION_STORAGE global variable
- [x] Supabase integration: Edge Functions deployed
- [x] PDF size limit: 10 MB
- [x] Fragment size limit: 500 characters
- [x] LRU cache configured: maxsize=10
- [x] Gunicorn config: 1 worker, max_requests=100

### Frontend Verification
- [x] HTML valid (no unclosed tags)
- [x] Modal using native `<dialog>` element
- [x] JavaScript variables: sessionId, parsedQuestions
- [x] No external dependencies (pure JS)
- [x] CSS colors: green (#d4edda) for correct answers
- [x] Responsive design implemented

### Database Verification
- [x] quiz_sessions table created
- [x] quiz_results table created
- [x] RLS enabled on both tables
- [x] Public access policies configured
- [x] TTL (expires_at) configured for 24 hours
- [x] Indexes created for performance

### Edge Functions
- [x] quiz_storage deployed (ACTIVE)
- [x] Handles: save_session, get_session, save_results
- [x] CORS headers configured
- [x] Error handling implemented

### Deployment Configuration
- [x] Procfile created and tested
- [x] render.yaml configured with memoryLimit: 512
- [x] Environment variables in .env
- [x] Python 3.11 specified

### API Endpoints
- [x] GET / - Main HTML page
- [x] POST /api/extract-text-from-pdf/ - PDF upload
- [x] POST /api/parse-quiz-html/ - HTML parsing
- [x] POST /api/process-quiz/ - Quiz processing

### Memory Usage Estimate
- Python runtime: ~100 MB
- FastAPI + deps: ~50 MB
- Buffers: ~30 MB
- LRU caches: ~20 MB
- Gunicorn: ~20 MB
- **Total: ~220 MB** (< 512 MB limit)

## 🚀 Deployment Steps

1. Connect GitHub repository to Render
2. Set environment variables:
   - VITE_SUPABASE_URL
   - VITE_SUPABASE_ANON_KEY
3. Deploy using Procfile configuration
4. Monitor logs for errors
5. Test with sample PDF and HTML test

## 📝 Post-Deployment Testing

1. Open app URL in browser
2. Upload PDF (test with 1-5 MB file)
3. Paste HTML test content
4. Click "Разобрать HTML"
5. Click "Сопоставить с лекцией"
6. Verify green highlighting on correct answers
7. Click on answer to view fragment in modal
8. Check that modal opens/closes properly

## ⚠️ Common Issues

### Issue: Illegal header value
- **Cause**: Empty SUPABASE_KEY
- **Solution**: Verify VITE_SUPABASE_ANON_KEY in .env

### Issue: 404 on process-quiz
- **Cause**: sessionId not stored/retrieved
- **Solution**: Check Edge Function logs, verify Supabase tables

### Issue: High memory usage
- **Cause**: LRU cache size too large or PDF too big
- **Solution**: Check PDF size (max 10 MB), verify cache settings

### Issue: Slow performance
- **Cause**: Large PDF or many questions
- **Solution**: Use smaller PDFs, optimize HTML parsing

## 📊 Performance Targets

- PDF upload: < 5 seconds (10 MB file)
- HTML parsing: < 2 seconds (100 questions)
- Quiz processing: < 5 seconds (per 100 questions)
- Modal open: < 500 ms
- Total memory: < 300 MB (with margin)

---

**Status**: Ready for Deployment
**Version**: 1.0
**Date**: 2025-11-27
