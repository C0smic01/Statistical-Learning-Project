/*
  # Create history table for storing emotion analysis results
  
  1. New Tables
    - `history`
      - `id` (uuid, primary key)
      - `input_text` (text)
      - `emotions` (jsonb)
      - `dominant_emotion` (text)
      - `created_at` (timestamp)
  
  2. Security
    - Enable RLS on `history` table
    - Add policy for public access
*/

CREATE TABLE IF NOT EXISTS history (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  input_text text NOT NULL,
  emotions jsonb NOT NULL,
  dominant_emotion text NOT NULL,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE history ENABLE ROW LEVEL SECURITY;

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