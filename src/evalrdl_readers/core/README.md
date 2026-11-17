# evalrdl_readers.core

A **reader** is a set of **inputs**. An input reads one kind of content and answers
**questions**. A **translator** is the ordinary function that turns bytes into answers. A
**landmark** is a directory found by a marker file, and everything else is addressed
rigidly relative to one.

What a reader may say is a closed list, every entry carries a written reason, and none has
a default. A mistake is refused loudly and early — at construction or at reader load,
before a file is opened. What a real folder gets wrong becomes a **finding**, which is
data you can enumerate, not an exception.

This is an implementation of the reader layer described in *Eval run data layer — the
reader layer* (RFC, 2026-09). It is standalone — nothing about any particular producer —
and it uses only the standard library. The readers that parse YAML use PyYAML
(`readers/common.py`): a hand-rolled subset parser, checked against PyYAML over the 190
YAML files in the corpus, silently lost keys on 130 of them.

```
libs/readers/
  src/evalrdl_readers/core/      the library (this directory); TUTORIAL.md teaches it
  src/evalrdl_readers/readers/   one reader per public producer, written in it
  tests/                         the core's tests: synthetic fixtures, no network, about a second
  tests/examples/                tiny_reader.py, one worked reader, and run/, the folder it reads
```

From `libs/readers/`:

```sh
python tests/examples/tiny_reader.py     # the worked reader
python -m pytest tests                   # the core's tests, which also run every python
                                         # block in TUTORIAL.md and ../README.md
```

---

## Changing a reader

A reader change is usually small once it is in the right place. Read before writing, in this
order:

1. **The decision record.** Search the project's knowledge graph (in EvalRDL, the `evalrdl-kg`
   skill) for the behaviour. A change usually widens or narrows a rule already written down:
   edit that record, or supersede it, instead of giving the rule a second owner.
2. **The code.** Find the expression that owns the behaviour and change it there. A second
   mechanism beside it, in another input or another reader, is the duplicate this order exists
   to prevent.
3. **The declaration docs.** Which input owns which fact decides where the change goes:
   [TUTORIAL.md](TUTORIAL.md), [the grammar](../../../docs/grammar.md),
   [constructs by topic](../../../docs/dsl.md), [lanes, keys and edges](../../../docs/lanes.md).
   An input states what its own records say; it never infers a role from another input's data.
4. **The shape, surveyed** (*Survey before you declare*): every variant the producer writes, not
   the folder in front of you.
5. **One change** carries the record, a check seen to fail without the fix (*Every defect is
   recorded and covered*), and the goldens rewritten and reviewed
   ([Testing a reader](../../../docs/testing.md)).

For example: a harness captures the policy's and a simulated user's model calls in one record,
both land in one `model_call` lane, and the lane's prompt sizes alternate between the two. A
rule already gives each endpoint (`model_ref.name`) its own lane, under a condition this record
shape does not meet. The fix widens that condition, in the input that reads the calls, to any
record naming more than one endpoint, and states each call's actor from the same record's
messages. The input that reads the conversation does not change.

## Every defect is recorded and covered

A defect found here -- by a test, by a reader author, or by somebody clicking a link -- is
written into the knowledge graph AND covered by a check that was **seen to fail on it**. The
preferred form is a test under `app/tests/reader_decls/`, because that is what runs over
every corpus folder.

Where the harness cannot observe the defect -- a browser-only symptom, a producer we hold
no example of, a race nobody can stage -- the record says so **in those words**, names the
weaker guard it got instead, and says plainly that it is unguarded if it got none. Silence
is read as coverage by the next person, and it is not coverage.

## Survey before you declare

A declaration written before the shape has been read is a guess, and this layer exists
because guesses reach a catalog looking like facts. Before writing an `Input`, a
`Decoder` or a `Contribute` over a shape nobody has read yet, survey it and keep the
survey:

- every key the records carry, with the count of records carrying each
- every field whose value is another record's id -- resolve it, and report **both** the
  resolved count and the dangling count
- every ordering and grouping field, and whether the order it states is the order the
  file is in

Enumerate; never sample, and never test only the first record of a file. MEASURED: eight
producer fields state a structure -- `parentUuid`, `sourceToolAssistantUUID`,
`isSidechain`, `promptId`, `turn.invocation_id`, `tool_call.invocation_id`, `turn_no`,
`ordinal` -- every one resolves, and none of them is read by any reader here. 12,558
parent pointers resolve with 0 dangling and 96 nodes branch, while the readers emit a
flat list.

A filter that selects which files to survey is itself a claim, and it must be measured
before it is trusted: a scan of that same tree reported it 25x too small because it
tested each file's FIRST record, and a trunk transcript opens with a record that carries
no parent.

A structure the survey finds and the reader does not carry gets a verdict -- TRUE, DEFECT
or NOT OURS -- never silence.

## Two minutes

```python
from evalrdl_readers.core import *

# 1. Where to look. A landmark is a directory found by a marker file.
RUN  = Landmark(name="run", marker="manifest.json", where=("", "artifacts"),
                expect=Card.ONE(because="one manifest per run"))
CASE = Landmark(name="case", marker="outcome.json", within=RUN, where=("cases/*",),
                expect=Card.AT_LEAST_ONE(
                    because="one directory per case IS the enumeration"))

# 2. What is stored. A question names a fact, its grain, how two answers combine,
#    the columns it fills, and what it means.
SCORE = Question(name="rollout.score", grain=Grain.ROLLOUT, merge=Merge.ONE,
                 lands="rollouts.reward", means="what this case scored")

# 3. The ordinary code. It gets exactly the files its own declaration matched.
def outcome(src: Sources):
    yield Answer(question="rollout.score",
                 value=src.doc(0).json()["score"],
                 key=(src.site.row,))   # .row, not .key — see TUTORIAL.md §4.6

# 4. The input: a name, a translator, what it reads, and how many answers to expect.
OUTCOME = Input(name="outcome", via=outcome,
                   reads=[At(CASE, "outcome.json")],
                   answers={"rollout.score":
                            Card.ONE(because="the producer's own record")})

# 5. The reader. `schema=` is the application's own {table: [column, ...]} mapping;
#    every `lands=` is resolved against it here, before a file is opened. Omit it and
#    nothing is checked, which is what a standalone reader wants.
READER = Reader(name="demo", landmarks=(RUN, CASE), inputs=(OUTCOME,),
                questions=(SCORE,),
                rows=(Rows(question=Grain.ROLLOUT, may_create=("outcome",),
                           because="a case directory IS the enumeration"),),
                schema={"rollouts": ["run_id", "task_id", "repeat", "reward"]},
                counts=Population.COMPLETED)   # which rollouts these rows are drawn from

out = READER.read("path/to/run")
out.value("rollout.score", ("case_0001",))   # -> 1.0
out.findings                                 # -> one UNREAD finding for manifest.json:
                                             #    no input reads it and no Ignore
                                             #    accounts for it. Nothing stays quiet.
```

The full worked reader — nested landmarks, two inputs answering one question, a
per-record decoder choice, two inputs contributing to one stream, an unresolvable edge, a fold to the run grain, and an ignore that says what it
is *not* hiding — is `libs/readers/tests/examples/tiny_reader.py`, reading
`libs/readers/tests/examples/run/`. Start there.

---

## The public API

Everything below is exported from `evalrdl_readers.core`. Every field, default and method is
in `libs/readers/docs/grammar.md`, generated from the code; this section adds the rules.

### Addressing

| | |
|---|---|
| `Landmark(name, marker, where, expect, within=None)` | a directory found by a marker file. `marker` is one path segment or an `AnyOf`; `where` is a tuple of rigid relative directory patterns (`""` = the enclosing landmark's own directory); `within` is the enclosing `Landmark` **object**; `expect` is how many sites per enclosing site. |
| `AnyOf(*names)` | one landmark reached by two spellings. Flat: nesting is refused. |
| `Says(marker, contains, because)` | a marker that must also contain a literal (at least three characters) in its first 256 KB. Pass it as a `Landmark`'s `marker`. |
| `At(landmark, pattern)` | a rigid relative file pattern anchored to a landmark. |
| `Ignore(landmark, pattern, because)` | a read that discards. Same shape as a read: anchored, rigid, reasoned. |
| `ROOT` | the landmark that is always there, the tree root. You rarely name it. |
| `Site.row` | the part of a site's address that identifies it: the segments that matched a `*`. **Use this for row keys.** `site.last_segment` is only the last path segment, so with `where=("*/*",)` every `<task>/repeat_0` collapses onto `repeat_0`. |
| `Site` | a resolved landmark instance: `.landmark`, `.path`, `.last_segment` (the directory's own name), `.row`, `.anchor`, `.matched`, `.chain`. |

**Pattern rules**, enforced at construction:

- a `*` never crosses a `/`; matching is segment by segment;
- `**` is allowed in `At` and `Ignore` only, must be **trailing**, and must **name the
  directory it covers**. `deliverables/**` is an envelope; `**/result.json` and `task_*/**`
  are refused;
- no absolute paths, no `..`, no backslashes;
- a `marker` is one path segment and may not contain `**`;
- a `where` pattern may not contain `**`.

### Counting and ranking

| | |
|---|---|
| `Card` | `NONE`, `ONE`, `AT_LEAST_ONE`, `ANY`. |
| `Population` | `COMPLETED`, `ATTEMPTED`, `PREPARED`: which rollouts a reader's rows are drawn from. `Reader(counts=)` has no default. |
| `Card.ONE(because=...)` | a `Card` member, called, is a count and the reason somebody believes it — that is what every `expect=` takes. Checked on answers **that carry a value**: a demanded count is a claim that a fact is *there*, and an answer of `None` is the absence of one. Never on how many sources spoke. A bare member carries no reason and is legal only in `Relate(cardinality=, to_card=)`, where the declaration already carries one. The same rule governs a settle: an `Order` ranks the sources that *stated* a fact, so a `None` that outranks a real value has settled nothing and is a `MISSING`. |
| `Rows(question, may_create, because)` | which inputs may **create** rows. `question` is one question's name, or a `Grain` — which says the same thing about every question at that grain. **Required for every question with a row key**, whatever its merge and however many inputs answer it: an `Order` picks a winner among sources that answered the same row and does nothing about rows only one of them names, and a `Contribute` says who contributes without saying which keys they may bring. Run grain is exempt because a run-grain answer has no key to invent. |
| `Order(question, order, keep_loser, because)` | who wins where two **sources** answer one question. `order` is a tuple of source names (see *Source names* below). `keep_loser` must be `True` or `False` — no default. |
| `Op` | `SUM`, `MEAN`, `MIN`, `MAX`, `COUNT`, `DISTINCT`. The vocabulary is closed. |
| `Fold(into, over, op, expect, because)` | N answers at one grain become one at the grain above. `expect` counts the subjects that did **not** answer (`Card.NONE` = do not publish a partial total, `Card.ANY` = publish it and name the shortfall); only those two mean anything here. |

### What is stored

| | |
|---|---|
| `Grain` | `RUN`, `ROLLOUT`, `EVENT`, `PART`, `CALL`. The ladder is fixed. |
| `Merge` | `ONE` (two inputs state the same fact; one wins, an order is required) or `UNION` (two inputs hold different parts of one fact; all land, and declaring an order is refused). |
| `Question(name, grain, merge, lands, means)` | a named fact we store. |
| `TABLES` | the ten tables the bundled corpus readers land in. A **default, not a law** — see below. |

`lands` is `"table"` or `"table.column"`. Pass the application's own
`{table: [column, ...]}` mapping as `Reader(schema=)` and every landing is resolved against
it at registration, over the real `Question` objects: a landing naming a column that does
not exist is a `Refusal` before a file is opened. `schema=None` checks nothing, and that
is the standalone default — a library that hardcodes one application's schema carries a
second schema, and the two drift.

Count landings **per column**, not per table: a landing naming a whole table passes
because the table exists while the column it needs may not. A whole-table landing is
checked again during the read, key by key (see `MISSING` below).

The bundled readers resolve against the real store schema plus `NO_HOME` in
`../readers/store_schema.py`: every fact the readers state and the store has no column for,
with a reason each. A typo still fails, because a name on neither list fails.

The landing is also checked injective across the whole reader: two questions landing in
one column is a startup error, and so is one question taking a table whole while another
names a column of it. Neither that check nor the grain ladder depends on the table list —
`DEPTH` and `PARENT` are keyed on grains — so a landing in a table with no rung on the
ladder costs nothing.

### Inputs and translators

| | |
|---|---|
| `Input(name, via, reads, answers, decoders=None, mints=())` | one named unit. `reads` must all be anchored to **one** landmark; the input runs once per site of it. |
| `Answer(question, value, key=None, input_key=None, keys={}, via=None, note=None, lane=None)` | what a translator yields. `lane` puts a stream row in a declared `Lane` (see *Attaching*). |
| `Unreadable(question, where, detail)` | the other thing a translator may yield: a record was there and could not be read. It fills nothing and is an `UNREADABLE` finding. |
| `Sources` | what a translator is handed: `.site`, `.doc(i)`, `.docs(i)`, `.has(i)`, `.all`, `.decode(record, question)`. There is no method that reaches another file. |
| `Doc` | one matched file: `.path`, `.relpath`, `.name`, `.size`, `.bytes()`, `.text()`, `.json(default=None)`, `.jsonl()`, `.bad_lines`. |
| `Key(space, because, optional=False)` | a key space this input is the source of. |
| `Decoder(name, because, decides=None)` / `Decoders(candidates, because)` | the candidate ways to read a record; with `decides=` predicates, `src.decode(record, question)` picks one. The choice is per **record**, always — the format is a property of the record, not the file, so every answer is stamped with the decoder that read it. `candidates` is a *set* of what may be used, not a ranking: where two could answer one `Merge.ONE` question, that is an `Order` over `input/decoder` names. |

A translator is `Callable[[Sources], Iterable[Answer]]`. It may only answer questions its
input declares, it may not name its own source, and it may not reach a file its `reads`
did not match.

### Attaching

| | |
|---|---|
| `Join(space)` | a join names a **key space**, never an input. That is why execution order cannot matter. |
| `Contribute(to, input, order_by, identity, because)` | this input adds rows to that stream; `order_by` and `identity` name the columns **in this input's own rows**. `identity` may be a tuple of columns, so a composite identity is a declaration rather than an f-string in a translator — where an interpolated missing field writes the literal `"None"` into the key. |
| `Lane(kind, order_by, identity, because, actor=None)` | one ordered sequence inside a stream row (a trace, a sub-agent's turns, a monitor), declared on `Reader(lanes=)`. An answer names its lane with `Answer(lane=)`; an undeclared kind is refused. Rows are deduplicated and numbered per lane, and `lane=None` is the one unnamed lane. |
| `Link(word, to, frm_input, to_input, on, cardinality, because, to_card)` | an edge between two rows of the stream `to`, matched on a key space: the end that mints `on`'s space (`mints=`) is named, and rows of the other name its key values under the relation `word` (`Answer(refs={word: [id, ...]})`, the input declaring `names=(Key(word, ...),)`). A row may name several, and one input may name by several words over one space without them mixing. `word` is what the edge is, a word of the application's vocabulary. Counts as on `Relate`: `cardinality` per `frm` row that carries the space, `to_card` per `to` row, both bare `Card`s, `to_card` required. Checked per row key and only where both inputs gave rows there: a source a folder does not hold is not checked, one it holds is. A name matching nothing is an `ORPHAN` (and, from the `frm` side, an edge with no far end, keeping the value it named, by digest past 256 characters). A pair is one edge however often it is named; a value reaching several rows links each, its reason saying one of how many. `inferred=True` (with `max_matches`, default 8): the space is derived from content (a text's key), not stated, so a miss is only counted, no counts are claimed, and a value reaching more than `max_matches` rows links none. |
| `Unlinked(word, input, because)` | a relation word the input's rows name others by that this reader joins to nothing, and why. Every word an input `names` is linked or declared unlinked. |
| `Relate(frm, to, on, cardinality, because, to_card)` | records an edge between two rows. `cardinality` is how many `to` rows each `frm` row should reach; `to_card` is how many `frm` rows each `to` row must receive, and `None` says that claim is not made. Both are bare `Card`s, and both are required — their reason is this declaration's own `because`. Omitting `to_card` is a `Refusal` naming the line to write, not a `TypeError` from the constructor. An end that cannot be named is recorded, never dropped. `Card.ANY` as the `cardinality` of a join the `frm` side does not mint is **refused**: every such row is the producer asserting that the thing it names exists, so a join resolving none of them would pass, forever. |

### Readers and output

| | |
|---|---|
| `Compare(left, right, must, because)` | two questions this reader answers, and the relation that must hold between them — `must` is `"equal"`, `"at_least"` or `"at_most"`. The **only** declaration that reads more than one question. It exists because of a measured read reporting `ok: True, quiet: True, findings: 0` with `run.n_expected = 2500` one column from `run.n_scored = 3`, publishing a pass@1 over 2,497 rollouts it did not have. No translator can do this: the two numbers come from two inputs reading two files, and a translator is handed its own input's files and has no method that reaches another's. A relation that does not hold is `MISSING`; a side that answered nothing is `UNCHECKED`. |
| `Reader(name, landmarks, inputs, questions, counts, orders, rows, folds, compare, ignored, attach, lanes, schema)` | validates everything at construction. |
| `Reader.detect(tree) -> (bool, [str])` | does this reader claim this folder, and if not, which declared condition failed. |
| `Reader.read(tree) -> ReadResult` | `tree` is a path, or any `Tree`. |
| `Mount(reader, within, where, expect, because, rollout=None, drop_run=None, wraps=False, inputs, orders, rows, attach, ignored, questions)` | another reader's tree inside this one. The mounted reader is written as if it read a folder of its own: its single root landmark has `where=("",)`, and the `Mount` places it at `where` under the host's `within`, `expect` sites per `within` site. Its landmarks, inputs and ignores are re-rooted; everything else is merged by name. For a question both answer, the host's sources rank first (`wraps=True`: the mounted reader's), with the loser kept. `rollout=Join(space)` makes each mounted rollout the host rollout whose `space` key equals the mount site's `row`. `drop_run` ignores the mounted reader's run-grain inputs, for a host holding one mount per rollout. The remaining fields are the seam's own declarations, which name both readers and so belong to neither; a seam `Order` or `Rows` replaces the merged one for its question. |
| `Reader.mount(*mounts) -> Reader` | this reader with every `Mount` placed in it, one flat `Reader` under this reader's name. Neither reader is changed. |
| `Dispatch(readers, because)` | a declared order between readers for where two claim one folder. The reader that needs less goes last. `.pick(tree)`, `.read(tree)`. |
| `ReadResult` | `.claimed`, `.rows`, `.losers`, `.edges`, `.findings`, `.sites`, `.hidden`, `.minted`, `.read_paths`; `.value(q, key=())`, `.stream(q, key=())`, `.keys_of(q)`, `.of_kind(kind)`, `.ok`, `.report()`, `.to_dict()`. |
| `Row` | `.question`, `.key`, `.value`, `.input`, `.input_key`, `.name`, `.source`, `.site`, `.at_path`, `.at_line`, `.keys`, `.lane`, `.position`, `.notes`. |
| `Edge` | `.kind`, `.space`, `.frm_question`, `.frm_key`, `.frm_name`, `.frm_lane`, `.frm_id`, `.to_question`, `.to_key`, `.to_name`, `.to_lane`, `.to_id`, `.reason`. `to_name` is `None` where the edge has no second end, and `.reason` says why. `frm_id`/`to_id` are each end's identity value where the edge was stated by identity (a row carrying it), else `None`. |
| `Finding` / `Kind` | `.kind`, `.question`, `.where`, `.detail`, `.source`, `.line()`. |
| `Refusal` | the one exception raised on purpose. |
| `Tree` / `LocalTree(path)` / `Index` / `Entry` | the bytes behind one small interface: `paths()` and `read()`, and deliberately nothing else. |

---

## The rules a declaration must satisfy

Every one of these is a `Refusal` at construction or at `Reader(...)`, with a message
naming the fix.

**Names and references**

- Two inputs with one name — the winning source is stored beside the value, so the
  record could not tell them apart afterwards.
- Two landmarks with one name; two questions with one name; two orders for one question.
- A read or an ignore anchored to a landmark the reader does not declare; a nested
  landmark whose parent the reader does not declare.
- An input answering a question the reader does not declare; a question nothing answers.
- An input whose `reads` span more than one landmark.

**Cardinality and reasons**

- Any `because`/`means` that is empty or blank.
- An `expect=` that is not a called `Card` — a bare member carries no reason; `Fold.expect`
  with a card outside `{NONE, ANY}`.
- A landmark with no `where`, or with a repeated `where`.

**Orders**

- Two sources can answer one `Merge.ONE` question and no `Order` is declared. The refusal
  prints the line to paste.
- An `Order` naming a source that cannot answer that question, or failing to name one that
  can — it must be a total order over the sources.
- An `Order` on a `Merge.UNION` question.
- `keep_loser` not declared `True`/`False`; an order of fewer than two names; a repeated
  name.

**Folds**

- A `Fold` that does not lift exactly one rung of the grain ladder.
- An **aggregate** fold (`SUM`, `MEAN`, `MIN`, `MAX`, `COUNT`) whose `into` question is
  also stated directly by an input — an aggregate is a property of a population and a
  statement is a property of a document, so an order would delete one of two questions
  sharing a name. `DISTINCT` is exempt, and may be ranked.
- A fold into a `Merge.UNION` question, into itself, or declared twice.

**Landings**

- A landing that is not shaped like `table` or `table.column`.
- With a `Reader(schema=)`: a landing naming a table the application does not have, or a
  column that table does not have. Without one, nothing is checked.
- Two questions landing in one slot; a whole table and a column of it.

**Key spaces** — four checks:

0. **a key space value is one id.** `Answer` refuses a tuple, a list, `None` or a blank
   string: every consumer looks the value up whole, so a set of references matches
   nothing, forever, while every structural check passes.
1. **every key space a join names has a minter.** Without it the join silently returns
   nothing.
2. **exactly one minter per key space.** Two makes the index order-dependent.
3. **a declared mint must actually mint.** This one cannot be done on paper: it watches a
   real read, and an input that declares a mint and emits no key for it is a
   `Kind.COVERAGE` finding (`result.minted[space] == 0`).

**Rows**

- A question with a row key that no `Rows` covers — by name or by its `Grain`. Every
  merge, and however few inputs answer it.

**Attaching**

- A `Relate` whose `cardinality` is `Card.ANY` on a key space the `frm` side does not
  mint. Every such row is a reference the producer wrote, and `Card.ANY` is not a weaker
  claim about them — it is no claim, which a join resolving nothing passes.
- A `Relate` with no `to_card`. `None` is a legal value and an omission is not.
- A `Contribute` to a `Merge.ONE` question, from an unknown input, from an input that
  does not answer that question, or declared twice.
- An input answering a `Merge.UNION` question with **no** `Contribute` — a stream needs
  the columns this input's rows carry their identity and their order in.
- A `Relate` whose target is at a higher grain than its source (it would not be settled
  yet).
- A `Link` whose ends do not both contribute to its stream, or where one end does not mint
  its space and the other name its word; a word an input `names` that no `Link` joins and no
  `Unlinked` declares; and, at read time, a row naming others by a word its input does not
  declare in `names`.

**Mounting**

- A mounted reader with more than one top-level landmark, or whose root looks anywhere
  but `""`; a mounted reader that itself mounts others.
- A `within` the host does not declare; a landmark name both readers declare.
- One question name with two grains, merges or landings.
- A `rollout` join on a key space no host input mints.
- A `drop_run` input answering run-grain and lower-grain questions at once.

**At read time** (these are `Refusal`s because they are bugs in the declaration or the
translator, not facts about the folder):

- A translator answering a question its input does not declare.
- A run-grain answer carrying a row key, or a lower-grain answer whose key is the wrong
  length, or a key that is not a tuple.
- One source answering one `Merge.ONE` row twice — a landmark resolving to more than one
  site wearing a question's clothes.
- An input with `decoders` yielding a record with no `via=`, or with an undeclared one;
  an input without `decoders` stamping one.
- A `Merge.UNION` row that is not a mapping, so its declared columns cannot be addressed.
- `src.doc(i)` where the pattern matched any number other than one.

---

## Reading a folder

```text
sites = resolve_landmarks(paths)                   # landmarks ARE sorted, by `within`
for grain in lowest_first:
    for inp in inputs_at(grain):                   # any order: no input ever
        for site in sites[inp.landmark]:           # sees another's output
            answers += inp.translate(files_at(inp.reads, site))
    for q, given in answers.by_question():
        if   len(given) == 1:   settled[q] = given[0]
        elif q.merge is ONE:    settled[q] = order.pick(given)      # losers recorded
        elif q.merge is UNION:  settled[q] = dedupe(given, q.identity)
    for attachment in attachments_at(grain):
        attachment.join(settled)                   # a miss is a row with a reason
    assign_positions(settled, q.order_by, tiebreak=(input, input_key))
    for fold in folds_from(grain):
        answers_at(grain + 1) += fold.apply(settled)
check_cardinalities(); check_every_column_is_claimed()
```

**Row names.** Every row is named `(input, input_key)`. `input_key` is the
producer's own id if the translator supplies one, and otherwise a number counted **within
that input's own output** — so a row's name never depends on who else read the folder.
`Row.name` renders it as `input/input_key`. The counter runs per
`(input, question, site)`, so a row's full address is
`(question, key, input, input_key)` — adding a case does not renumber another
case's rows. `position` (`event_idx`) is assigned last, for display, per `(question, key)`.

**Source names.** The label stamped on a value so it says where it came from. Most are an
input's name. A decoder that won one record is stamped `input/decoder`. A folded
total is stamped `fold:<over>`. `Order` ranks these strings.

**Two properties, and they are the point.** Reading the same folder twice gives the same
output, because every input to the sort is either declared or deterministic — which is why
a tie falls back to `(input, input_key)` rather than to whatever order the inputs
ran in. And adding an input does not rename the rows that were already there. Both are
tested, including a test that runs every permutation of the inputs and asserts one
output.

**Processes.** With `EVALRDL_READ_PROCESSES` of 2 or more, inputs that share at least
`EVALRDL_READ_PROCESS_MIN_BYTES` of files are dealt over that many child processes; the
result is the one threads give. Children start from a forkserver, never by forking the
reading process: a fork copies every lock its other threads held, held by threads the child
does not have. A child imports the reader from the module that holds it, so only a reader
that is some module's attribute reads in processes; any other reads in threads. Children
import the program's `__main__` as `multiprocessing` does, so a script that reads in
processes keeps its work under `if __name__ == "__main__":`. `EVALRDL_READ_START_METHOD=fork` is taken only while the process has one OS thread, and is
refused (a `read_fork_refused` log event) otherwise. If no subset finishes for
`EVALRDL_READ_STALL_S` seconds (7200), the children are terminated and the read raises
`procs.ReadStalled`. A child that exits without its subset, most often killed for memory,
sends its group back to threads (a `read_child_lost` log event): one parse of the shared files
instead of one per child.

### Findings

```python
for f in result.findings:
    print(f.line())
result.of_kind(Kind.ORPHAN)
```

| `Kind` | when |
|---|---|
| `MISSING` | a declared cardinality was not met: a landmark, an answer, an answer that carried no value, a `Relate`'s `cardinality` or `to_card`. Also a declared `identity`/`order_by` column absent from a contributed row, a `Merge.UNION` identity collision whose values differ, a settle won by a `None` while another source carried a value, and a row key that no column of its whole-table landing has a home for. |
| `CONTESTED` | two sources said different things about one thing and an `Order` picked one; a file hidden — or claimed — by two declarations. Two rows of a `Merge.UNION` stream claiming one identity with different values is **not** this: `Merge.UNION` refuses an `Order`, so nothing settled it and the loser's value is gone — that is `MISSING`. |
| `COVERAGE` | a fold published a partial total or refused to publish one; **a declared mint minted nothing**. |
| `ORPHAN` | a join matched nothing and nothing was demanded: a `Relate` declaring `Card.ANY`. Where a **declared** cardinality is unmet the severity follows what was declared, and it is `MISSING`. |
| `UNREAD` | a file no input reads and no `Ignore` accounts for. An unexamined file stays loud. |
| `UNBACKED` | a source described a row no enumerating source produced. |
| `UNREADABLE` | a record a translator was handed and could not read — yielded as an `Unreadable`, or counted by the `Doc` that handed out the parse error and not reported back. A whole document that failed to parse counts: `json()` records its failure the way `jsonl()` records a bad line, so a truncated file is no longer indistinguishable from an empty one. |
| `UNCHECKED` | **a declared check that never ran, and what it would have checked.** A landmark declaring `Card.ANY` that resolves to zero sites takes every count underneath it with it: measured, **286 written `Card.ONE`/`Card.AT_LEAST_ONE` claims never evaluated across 37 of 53 folders, against 285 `MISSING` findings actually emitted.** Deliberately **not** in `LOSS` — an absent directory the landmark said in writing may be absent holds no rows to have lost, and putting it in `LOSS` would make `Card.ANY` on a landmark mean nothing. It costs `quiet`, which is the point: what was missing was never the verdict, it was the sentence. Also raised where a `Compare` had only one side to work with. |

Findings come out in one fixed order, so two reads of one folder produce identical
complaints. `result.ok` is `claimed and no finding whose kind is in `LOSS`` — UNREAD, MISSING,
COVERAGE, UNREADABLE, UNBACKED. `CONTESTED`, `ORPHAN` and `UNCHECKED` do not fail a read:
the first two mean an `Order` or a recorded join miss did its job, and the third says
something was never looked at rather than that something went wrong; `result.quiet` is the stricter "nothing at all to
report". A read with findings still has rows.

---

## Conventions this library fixes, where the RFC left a choice

The RFC is the specification. Where it was silent or ambiguous, this implementation picked
the reading that fails loudest, and each choice is written here.
1. **A row key is a tuple, and the key ladder mirrors the grain ladder.** A `Grain.RUN`
   answer has key `()`; `ROLLOUT` has one component; `EVENT` two; `PART`/`CALL` three. The
   length is checked against the question's declared grain on every answer — this is
   "grain is checked against the answer, not guessed from the name". A fold groups its
   subjects by `key[:-1]`, which needs no further declaration.
2. **A fold's subject population is what the read produced**, not what a schema says
   should have been there: a subject at grain *g* is any key some input answered
   something about at grain *g*. If only the folded question itself is asked at that grain,
   there are no gaps to report by construction — declare a second question at that grain
   (a score, a stream) if you want the shortfall counted.
3. **`identity` and `order_by` live on `Contribute`**, per input, because that is where
   the RFC writes them and because "the sidecar writes its own events, with its own ids and
   its own clock". Consequently **every input answering a `Merge.UNION` question must
   have a `Contribute`** — there is no default column name to fall back to.
4. **A question lands in one place, written as a string, and the string is resolved.**
   `lands` names a table or a column of one, and the landing is checked injective across
   the reader. Whether the table and the column exist is resolved against the
   `{table: [column, ...]}` mapping the application passes as `Reader(schema=)`, at
   registration, over the real `Question` objects — which is the only place it can be
   done: two of the bundled readers build a question as `type(Q.RUN_BENCHMARK)(...)`, so
   no scan of the source can see them. With no `schema=` nothing is checked, and that is
   a stated default rather than a silent one.
5. **An input runs once per site of its landmark, full stop**, and its answers are
   bucketed by the grain of the question each one answers. The RFC's loop says
   `inputs_at(grain)`; an input that answers at two grains (a summary file giving both
   `run.model` and `rollout.score`) is ordinary, and this reading admits it without letting
   any input see another's output.
6. **A `Relate` row carrying no key in the join's space is 0 matches.** A `Relate`
   declares a `cardinality`, so with `Card.ONE` a keyless row *is* reported — which is the
   RFC's deliverable-with-no-event case. **The severity of
   that report follows what was declared**: an unmet demand is `MISSING`, and only
   `Card.ANY` — which demanded nothing — leaves an `ORPHAN`.
7. **Agreement is not a finding.** Where an order settles two sources that said the same
   thing, the loser is recorded (if `keep_loser`) and nothing is reported. Only a
   disagreement the order had to settle is a `CONTESTED` finding.
8. **A miss is recorded and a decoder is chosen per record — always, with nothing to
    declare.** The RFC gives one legal value for each, so there is no choice to write down:
    a join that matches nothing produces a finding and a note on the row, and every answer
    is stamped with the decoder that read that record.
9. **A key space names one thing.** A tuple is hashable, so a translator wanting a
    one-to-many reference can hand three ids over as one key-space value and nothing
    structural objects — and every consumer looks the value up whole, so it matches
    nothing, ever. Measured on one folder: 1,243 edges, 0 resolved, 1,243 findings, `ok`
    True. `Answer` refuses it: one answer names one thing, and a set of references is
    several answers carrying the same row key.
10. **A count counts answers that carry a value.** `Card.ONE` used to count *answers*, so
    `Answer("rollout.score", None)` satisfied it completely. Measured over 53 folders:
    1,325 of 29,460 rows held `None`, and 19 folders reported `ok` while doing it. A
    producer that legitimately writes nothing is `Card.ANY`; a translator that cannot read
    a record says `Unreadable`.
11. **An input's declared count is checked at every site of its landmark, even one where
    its files are absent.** "An input that finds nothing contributes nothing — normal, not
    an error" is what `Card.ANY` says; `Card.ONE` on an input whose file has moved is
    exactly the opening defect this design exists to catch, so it reports. A count counts
    answers per row key, per site.

---

## What is here and what is not

Implemented: landmarks (nesting, `AnyOf`, nearest-wins, per-anchor cardinality, detection),
rigid patterns and envelopes, inputs and translators, counted answers, `Order` with
recorded losers, the six `Fold` ops, `Ignore` with double-hide and read-and-hide detection,
unread accounting, `Question`/`Grain`/`Merge`, `lands` injectivity **and resolution
against a registered schema**, `Contribute`/`Relate`/`Join`, `Lane`, `mints`/`Key` and
all four key-space checks, `Decoders` with decoder-stamped sources, the merge algorithm with
stable `(input, input_key)` naming and positions assigned last, `Dispatch`, and
findings as data.

Not implemented, deliberately:

- **Materialising the tables.** A reader here produces settled rows, streams, edges and
  findings, named by question and key. Turning those into tables is the schema layer's
  job, and the RFC treats the table list as a schema decision. `lands` is declared and
  checked for injectivity; it is not executed, and the columns it names are resolved there.
- **Group head tables.** There is no membership landing and no `group_members` table: it
  had no writer and no store here to write into, so a question about a group emits ordinary
  rows like any other. Edges from `Relate` are returned as `Edge` objects.
- **The RFC's seven open questions** — the credential-scanner predicate, dated falsifiable
  reasons, whether a null counts as an answer, detection failure as more than a boolean,
  markers that read file contents, whether anything treats `event_idx` as stable, and
  unarchived data. One of them is now decided: **a null does not count as an answer.** A
  demanded count is a claim that a fact is there, and an answer of `None` is the absence of
  one — measured, 1,325 of 29,460 rows held `None` and 19 folders reported `ok` while doing
  it. That holds at the settle too: an `Order` ranks the sources that *stated* a fact, so a
  `None` winning over a real value is a `MISSING`, not a `CONTESTED`. The rest are open in
  the RFC and not decided here; `Row.position` is still a display position with nothing
  depending on it.
- **`counts=` is required and nothing verifies it.** `Population` is ordered — COMPLETED
  inside ATTEMPTED inside PREPARED — so a claim about one is falsifiable against a claim
  about another over the same folder, and against nothing inside a single read. Measured: 7
  of the 53 corpus folders are claimed by two readers, and in none of them does the reader
  declaring the larger population produce the larger count; in 6 it produces 0 against 4 to
  432. The only object holding more than one reader is `Dispatch`, and whether its `read`
  should read every claimant to compare rather than only the winner is a cost decision
  nobody has made. Until it is, `ReadResult.counts` is published for the application to
  compare and `counts=` is a declaration nothing checks. Do not build on it.
- **A declared type for a value.** `Reader(schema=)` now carries the real columns, so
  `Answer("run.benchmark", {"a": 1})` landing in a text column *could* be caught. It is
  not: mapping a store's declared types onto acceptable Python values is a judgement with
  a long tail — ints into float columns, bools into ints, `datetime` against ISO strings,
  list columns — and half a coercion is worse than none. It is also not expressible at the
  question level as things stand: a `Merge.UNION` question's value is a whole row mapping
  while its `lands` names one column, so for those questions there is no value-to-column
  correspondence for a type check to use.
- **A byte budget / truncated-read record.** The prototype the RFC was measured on capped
  reads over a network transport and recorded truncation. `Tree.read(path, limit)` takes a
  limit, and `Doc.bytes(limit)` passes one, but nothing in the runtime imposes a budget or
  reports a partial read — there is no network transport here to need one.

- **A per-column declaration for a combined row, so a null cell can be reached.** This is
  the largest thing this round found and did not build, and the reason is worth writing
  down rather than hiding in a backlog.

  A `Merge.UNION` value is a **whole mapping** — one event, one part, one tool call — and
  every check in this library operates on the row, not on its cells:

  ```text
  STORED rollout.events ('c0',) = [{'kind':'msg','role':None,'started_at':None,'content':None}]
  findings : []   ok : True   quiet : True
  ```

  Three of four cells empty and nothing said. One author measured **180 null cells across
  two held-out folders and zero in training**, by hand, after the fact. This is where the
  conversation data lives — events, parts, tool calls, the highest-volume tables in the
  store — and it is the same coarse-unit mistake as the three that *were* fixed this
  round: the row exists, so the check passes, while the cells inside it are empty.

  It was **not** fixed, and a blanket "no null cells" rule is why: `events.content` is
  legitimately null for a tool-call event, so the rule would be wrong on the first real
  folder and would be deleted by the first author who met it. Knowing which cells may be
  empty needs a per-column declaration, which is the typed half of the `Owns` idea the RFC
  already records as unbuilt, and which has no home in this vocabulary today.

  The near miss is `Contribute`, which already names two columns of a contributed row.
  Adding `required=(...)` there would reach the cells — and it would have to be a
  **required** parameter to mean anything, because a check that is available and not
  required is a suggestion and the measured compliance rate with the suggestions in this
  library is near zero. That is 37 `Contribute` call sites in the shipped readers, each
  needing a written claim about which of its columns a producer always fills — claims
  nobody here has the evidence to write. Writing 37 plausible sentences would produce
  exactly the thing the next entry in this list is about.

  So: measured, named, unbuilt, and the reason is in writing. A half-built check here
  would have been the sixth instance of the pattern this round existed to fix.

- **Any verification at all of a `because=`.** Every declaration carries a written reason
  and the only check that runs on all of them is that it is not empty. Measured across the
  shipped readers, the example and the tutorial: **9,202 words of `because=`/`means=` prose
  across 508 declarations, verified only for non-emptiness.** That is not a gap to be closed — prose
  is not checkable — but it has a cost, and this round paid it in the worst possible
  place. `TUTORIAL.md` told the learner to hide the file holding all the metrics:

  ```python
  Ignore(NS_RUN, "results.yml", because="the resolved command line, already ...")
  ```

  It is the resolved command line **and** the published metric tree: 8,038 numeric leaves
  under `results:` across the 15 such files in `corpus2/train`, and the one file both
  `nemo-*` producers write identically — the only place a shared translator could live.
  A written reason that was wrong, hiding real data, passing every check, in the document
  that argues declarations beat code. It is fixed in the page; nothing could have caught
  it.

  The conclusion is not "check the prose". It is the argument for keeping the vocabulary
  small enough that the **structure** carries the claim and the prose only explains it.

- **A blank or `"None"`-shaped key component, refused.** Proposed and deliberately left
  open: an f-string interpolating a missing field writes the literal `"None"` into a row
  key, which is the fingerprint of a real bug — and is also a legal id some producer will
  eventually write. `Answer` already refuses a blank *key-space* value for that reason;
  extending it to row-key components is a different trade, on a different failure rate,
  and it is noted here rather than guessed at.

---

## What `ok` actually claims

Worth saying plainly, because nine of the nine attacks this library has survived lived in
the gap between what `ok` means and what people read it as.

`ok` means: **every file in the tree was named by some declaration, every declared count
was met, every declared join and fold came up full, and every declared `Compare` held.**
Those are four real claims and most pipelines have none of them.

**"Correct" is not one of them.** A translator that reads the wrong field, converts a
verdict with a helper that answers a slightly different question, or quietly drops the
records it does not recognise, produces a green read — and every one of those is a real
reproduction from the campaign, not a hypothetical. The declarations are checked; the
translator is not, and cannot be, because it is ordinary Python and that is the whole
point of it being ordinary Python.

The three things that narrow the gap are, in order: run the reader on a **second folder
from the same producer**, count something **outside the library** and compare, and write
the count you believe into a `Compare` so the library can check it for you next time.
