/*
  # Update history table and policies

  1. Table Changes
    - Create history table for storing emotion analysis results
    - Columns:
      - id (uuid, primary key)
      - input_text (text)
      - emotions (jsonb)
      - dominant_emotion (text)
      - created_at (timestamptz)
  
  2. Security
    - Enable RLS
    - Add policies for public read and insert access
*/

CREATE TABLE IF NOT EXISTS history (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  input_text text NOT NULL,
  emotions jsonb NOT NULL,
  dominant_emotion text NOT NULL,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE history ENABLE ROW LEVEL SECURITY;

DO $$ 
BEGIN
  -- Drop existing policies if they exist
  DROP POLICY IF EXISTS "Public can read history" ON history;
  DROP POLICY IF EXISTS "Public can insert history" ON history;
  
  -- Create new policies
  CREATE POLICY "Public can read history"
    ON history
    FOR SELECT
    TO public
    USING (true);

  CREATE POLICY "Public can insert history"
    ON history
    FOR INSERT
    TO public
    WITH CHECK (true);
END $$;