/*
  # Create history table for emotion analysis

  1. New Tables
    - `history`
      - `id` (uuid, primary key)
      - `user_id` (uuid, references auth.users)
      - `input_text` (text)
      - `emotions` (jsonb)
      - `dominant_emotion` (text)
      - `created_at` (timestamptz)

  2. Security
    - Enable RLS on `history` table
    - Add policy for authenticated users to read their own data
    - Add policy for authenticated users to insert their own data
*/

CREATE TABLE IF NOT EXISTS history (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users NOT NULL,
  input_text text NOT NULL,
  emotions jsonb NOT NULL,
  dominant_emotion text NOT NULL,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own history"
  ON history
  FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own history"
  ON history
  FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);