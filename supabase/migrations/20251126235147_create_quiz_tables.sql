/*
  # Create Quiz Helper Tables

  1. New Tables
    - `quiz_sessions` - Хранит сессии пользователей с текстами лекций
      - `id` (uuid, primary key)
      - `session_id` (text, unique) - идентификатор сессии
      - `lecture_text` (text) - текст извлеченной лекции
      - `created_at` (timestamp)
      - `expires_at` (timestamp) - время истечения сессии (24 часа)
    
    - `quiz_results` - Кэш результатов обработки вопросов
      - `id` (uuid, primary key)
      - `session_id` (text) - связь с сессией
      - `results_json` (jsonb) - результаты обработки (сжато)
      - `created_at` (timestamp)
      - `expires_at` (timestamp)

  2. Security
    - Enable RLS on both tables with public read/write policies (simplified for public access)
    - Add TTL-based cleanup via expires_at column
*/

CREATE TABLE IF NOT EXISTS quiz_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id text UNIQUE NOT NULL,
  lecture_text text NOT NULL,
  created_at timestamptz DEFAULT now(),
  expires_at timestamptz DEFAULT (now() + interval '24 hours')
);

CREATE TABLE IF NOT EXISTS quiz_results (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id text NOT NULL REFERENCES quiz_sessions(session_id) ON DELETE CASCADE,
  results_json jsonb NOT NULL,
  created_at timestamptz DEFAULT now(),
  expires_at timestamptz DEFAULT (now() + interval '24 hours')
);

CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON quiz_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON quiz_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_results_session_id ON quiz_results(session_id);
CREATE INDEX IF NOT EXISTS idx_results_expires ON quiz_results(expires_at);

ALTER TABLE quiz_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE quiz_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public access to quiz_sessions"
  ON quiz_sessions FOR ALL
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Public access to quiz_results"
  ON quiz_results FOR ALL
  USING (true)
  WITH CHECK (true);
