Fixed
A case or `-`/`_` variant of a third-party algorithm plugin's name that failed
to load (`algorithms show ACME-Method` for a broken `acme_method`) now reports
the plugin's load error instead of "unknown algorithm" (#698).
