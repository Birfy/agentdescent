Changed
An `--algorithm` / `algorithm:` kind mismatch (in `init`, `plan`, `evolve`,
campaigns and tournaments) now says which named algorithms do apply to that
kind (`agentdescent algorithms list --kind <kind>`), suggests targeting a skill
directory's single `SKILL.md` as `--kind text` when the algorithm runs on text,
and points to the default loop when nothing applies.
