Fixed
`rewards` reads numbers as they are written (#288). `last_number`,
`numeric_close` and `command_scorer` used to take `1.5e-3` as `-3`,
`6.02e23` as `23`, `.5` as `5`, `pages 3-4` as `-4` and `1,2,3` as `123`;
the parser now understands exponents and leading decimal points, groups
commas only in threes, and treats a minus glued to a preceding digit or
letter as subtraction. An overflowing figure (`1e999`) is no longer read as
`inf`, so `command_scorer` raises `GraderError` instead of clamping it to
`1.0`.
