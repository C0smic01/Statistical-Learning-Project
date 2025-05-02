/*
  # Add DELETE policy for history table

  This migration adds the missing DELETE policy for the history table.
  Without this policy, delete operations are rejected by Row Level Security (RLS).
*/

BEGIN;
  -- For authenticated users (matches the original table design)
  CREATE POLICY "Users can delete own history"
    ON history
    FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);
  
  -- For public access (if you've modified the app to work without authentication)
  CREATE POLICY "Public can delete history"
    ON history
    FOR DELETE
    TO public
    USING (true);
COMMIT;