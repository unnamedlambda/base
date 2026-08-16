/-- Shared by both packages, so editing it makes both go stale at once and
their build scripts run together. That is the case the lock exists for. -/
def Common.bump : Nat := 0
