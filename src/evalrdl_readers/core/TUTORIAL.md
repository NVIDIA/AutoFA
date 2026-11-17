# Writing a reader

You have never seen this library before. That is the only assumption on this page.

By the end you will have written a reader for a real producer folder, and you will know
why every line of it had to be there. **Every fenced `python` block on this page runs, in
order, in one namespace**: `tests/test_tutorial.py` executes them. Paste
them in the same order.

§12 reads folders from the corpus checkout (`READERDSL_CORPUS`, default `~/code/readerdsl`),
by paths relative to it: run from that directory to get the output shown there.

---

## 1. What this library is for

An **evaluation run** is a program that asks a model many questions — a benchmark, an
agent fixing bugs, a judge scoring answers — and writes what happened into a folder. A
**producer** is the tool that wrote the folder. There are many producers, each writes a
different folder shape, and each changes its shape without telling anyone.

We want those folders as rows in a database, so people can query them.

A **reader** is the thing that turns one producer's folder into rows. You write readers.
This page teaches you how.

Here is the bug the library exists to prevent. One producer moved two files between
versions:

```text
version 46                            version 47
  task__X/                              task__X/
    artifacts/traces/*.jsonl              agent/artifacts/traces/*.jsonl   <- moved
    verifier/criteria/*/reward.json       verifier/report.json             <- replaced
```

The old reader still ran. It looked for the transcript, found nothing; looked for the
score, found nothing; wrote the run down as *unscored*. The score was in the folder the
whole time. Nothing crashed, nothing was logged. **A failed search looked exactly like an
empty folder.** Months of results were wrong and nobody knew.

So the one promise: *a producer changing its output can never quietly produce a wrong
table. Either the reader works, or it says what it can no longer find.*

The cost of that promise is that this library refuses things. It refuses more than you
will expect, it refuses at construction time before any file is opened, and it never
guesses a default. §5 is what it reports about a real folder, and §6 is the catalogue of
the ten checks most likely to refuse you. Read the refusal messages — each one names the
line to write instead.

---

## 2. The shape of a reader

Reading a folder is three jobs that usually get mashed into one class:

| job | question | who does it here |
|---|---|---|
| **find** | which files are the ones I want? | a declaration |
| **interpret** | what do these bytes mean? | your ordinary Python |
| **conform** | which column does this value fill? | a declaration |

They change for unrelated reasons: the producer moves a file (find changes), the producer
adds a field (interpret changes), the database gains a column (conform changes). One class
doing all three means any of them can silently break the others.

So: **declarations find and conform; your code only interprets.** Your code never opens a
path, never picks a column, never picks a winner between two files. It is handed the bytes
its declaration matched, and it yields values.

---

## 3. The vocabulary

`evalrdl_readers.core.__all__` exports **53 names**. **31 of them are names you type** when writing a
reader; the other 22 are result objects, tree objects and fixed lists the runtime hands
back to you. Here are the 31, so you know what exists before you meet it:

| group | names |
|---|---|
| finding files | `Landmark`, `At`, `Ignore`, `AnyOf`, `Says` |
| the unit of reading | `Input`, `Sources`, `Doc`, `Translator` |
| what a translator yields | `Answer`, `Unreadable` |
| what you store | `Question`, `Grain`, `Merge` |
| how many | `Card`, `Population` |
| settling disagreement | `Order`, `Rows` |
| joining | `Key`, `Join`, `Contribute`, `Relate` |
| several sequences in one row | `Lane` |
| lifting a number | `Fold`, `Op` |
| comparing two answers | `Compare` |
| two formats in one file | `Decoder`, `Decoders` |
| assembly | `Reader`, `Dispatch` |

`Compare` is the one check that reads two questions; §12.3 is why it exists.

`Translator` is a type alias, not something you call, so in practice you type 30 of them.
§15 lists the other 22 and says what each is for, including the ones this page never
needs.

Older code may use `Provider` (now **`Input`**) or `Expect(Card.ONE, because=...)` (now
**`Card.ONE(because=...)`**); neither exists any more.

---

## 4. Your first reader

### 4.1 A folder to read

Run this. It builds a small folder that has, deliberately, one of everything that goes
wrong.

```python
from pathlib import Path
import shutil

LESSON = Path("/tmp/evalrdl_readers.core-lesson")
shutil.rmtree(LESSON, ignore_errors=True)
(LESSON / "cases/case_0001").mkdir(parents=True)
(LESSON / "cases/case_0002").mkdir(parents=True)
(LESSON / "manifest.json").write_text('{"benchmark": "demo", "model": "m-1"}')
(LESSON / "summary.json").write_text(
    '{"cases": {"case_0001": 1.0, "case_0002": 0.0, "case_0003": 0.5}}')
(LESSON / "cases/case_0001/outcome.json").write_text('{"score": 1.0}')
(LESSON / "cases/case_0002/outcome.json").write_text('{"score": null}')
(LESSON / "cases/case_0001/turns.jsonl").write_text(
    '{"id": "e1", "ts": 1}\n{"id": "e2",\n')
print(sorted(p.relative_to(LESSON).as_posix() for p in LESSON.rglob("*") if p.is_file()))
```

That folder is:

```text
/tmp/evalrdl_readers.core-lesson/
  manifest.json              benchmark, model
  summary.json               a rollup naming three cases
  cases/case_0001/
    outcome.json             {"score": 1.0}
    turns.jsonl              two records, the second cut in half
  cases/case_0002/
    outcome.json             {"score": null}      <- scored, or not scored?
```

Three cases are named in `summary.json`; two directories exist. One score is `null`. One
transcript is truncated. Every one of those is a finding, and you will see all three.

### 4.2 Landmarks: where to look

A **landmark** is a *kind of directory*, found by a **marker** — a file whose presence
means "this directory is one of those".

```python
from evalrdl_readers.core import *

RUN = Landmark(name="run", marker="manifest.json", where=("",),
               expect=Card.ONE(because="the producer writes one manifest per run"))

CASE = Landmark(name="case", marker="outcome.json", within=RUN, where=("cases/*",),
                expect=Card.AT_LEAST_ONE(
                    because="one directory per case IS the enumeration of cases"))
print(RUN, CASE)
```

Four separate ideas in there:

- `marker="outcome.json"` — a directory is a `case` if it holds a file with that name. A
  marker may be a glob: `marker="eval-*.json"` works. It is **one path segment**; where to
  look is `where`.
- `within=RUN` — look for cases inside a run, nowhere else. Landmarks nest, and that
  nesting is how you describe a *shape* rather than a file.
- `where=("cases/*",)` — and only in these places, relative to the enclosing landmark.
  `*` matches one path segment and never crosses a `/`. `""` means the enclosing
  landmark's own directory. A `where` may be several segments deep: `("*/*",)` is normal.
- `expect=` — how many there should be, **per enclosing directory**, with the reason
  attached. This is what makes detection a check instead of a guess.

**There is no recursive search.** You cannot write `where=("**",)` or
`At(RUN, "**/outcome.json")` to mean "find it somewhere". It is refused:

```python
try:
    Landmark(name="case", marker="outcome.json", within=RUN, where=("**",),
             expect=Card.ANY(because="somewhere under the run"))
except Refusal as e:
    print(e)
```

On one real tree, a recursive search for a config file matched **151 files, 75 of them
container fixtures, at three depths that meant three different things**. You say where
things are. When the producer moves them, you find out.

**Nearest wins.** A qualifying directory inside another qualifying directory of the same
landmark is not a second site. That is what makes `task__X/result.json` the case and
`task__X/agent/result.json` not a second one.

**`because=` is not a comment.** Every declaration takes one, an empty one is refused
before a file is opened, and it is what the failure message says when the count is wrong:

```python
try:
    Card.ONE(because="")
except Refusal as e:
    print(e)
```

### 4.3 Questions: what you are storing

A **question** is a named fact you want in a table.

```python
SCORE = Question(name="rollout.score", grain=Grain.ROLLOUT, merge=Merge.ONE,
                 lands="rollouts.reward", means="what this case scored")
print(SCORE)
```

- `name` — free text; the convention is `<owner>.<fact>`. **The prefix is not the grain.**
  `rollout.events` can be an `EVENT`-grain question about a rollout's transcript. Only
  `grain=` counts.
- `grain` — how big the thing it is about is. The ladder is fixed and five rungs long:
  `RUN` > `ROLLOUT` > `EVENT` > `PART` and `CALL`. A **rollout** is one case: one attempt
  at one task. An **event** is one turn inside it.
- `merge` — what happens when two inputs both answer it. `Merge.ONE`: they state the same
  fact, one wins, and you must rank them. `Merge.UNION`: they hold different parts of one
  list, and all of them land.
- `lands` — which column this fills, written `table` or `table.column`.
- `means` — what it means, for a human who finds the column in a year.

`grain` maps to a key length, and it is checked against every answer, not guessed:

```python
print({g.name: DEPTH[g] for g in Grain})
```

A `RUN`-grain answer carries no key. `ROLLOUT` carries one component, `EVENT` two, `PART`
and `CALL` three. Get the length wrong and it is a `Refusal`, not a wrong row.

### 4.4 `lands=` is resolved, not believed

Hand the `Reader` your application's real `{table: [column, ...]}` mapping as `schema=`,
and every landing is resolved against it at registration — before a file is opened:

```python
STORE = {"runs": ["benchmark", "model"], "rollouts": ["reward", "stop_reason"]}
try:
    Question(name="rollout.score", grain=Grain.ROLLOUT, merge=Merge.ONE,
             lands="rollouts.", means="what this case scored")
except Refusal as e:
    print(e)
```

That is the **shape** check — `table` or `table.column`, neither half blank, at most one
dot — and it happens on the `Question`, which knows nothing about your database. The
**existence** check happens on the `Reader`, because that is where `schema=` arrives, and
you will see it fire in §4.8.

This matters more than it sounds: before the check existed, most landings in the shipped
readers named nothing, and every affected read reported success. Count **per column**: a
landing naming a whole table passes while the column it needs is missing.

Most misses are not typos. `rollouts.score` is called `reward` in the store;
`runs.started_at` is called `run_ts`; and many name facts the store has **no column for at
all**.

When that happens to you, do not delete the declaration. `../readers/store_schema.py`
keeps a `NO_HOME` mapping — every column the readers need and the store lacks, one written
reason each, merged into the registered schema. A typo still fails, because a name on
neither list fails. Leave `schema=None` and nothing is checked, which is right for a
standalone reader and is why the library carries no table list of its own.

### 4.5 The translator: your ordinary code

A **translator** is a plain function. It is handed the files its input's declaration
matched, and it yields answers. It is the only part of a reader that is not a declaration.

```python
def outcome(src):
    yield Answer(question="rollout.score",
                 value=src.doc(0).json()["score"],
                 key=(src.site.row,))
print("translator:", outcome.__name__)
```

`src` is a **`Sources`**: one site, and the files this input's own patterns matched there.
It has no method that reaches any other file.

| on `Sources` | |
|---|---|
| `src.doc(0)` | the single file pattern 0 matched. **Refuses on any other count, including zero.** |
| `src.docs(0)` | every file pattern 0 matched, in path order |
| `src.has(0)` | did pattern 0 match anything? Guard `doc(0)` with this. |
| `src.first(0)` | the first match, or `None` |
| `src.all` | every file from every pattern |
| `src.site` | the directory this run of the translator is about |

And a **`Doc`** is one matched file. This list is closed — there is no `open()`:

| on `Doc` | |
|---|---|
| `.json(default=None)` | parsed JSON. **Returns `default` on a parse failure; it does not raise — and records the failure.** |
| `.jsonl()` | one `(line_number, record, error)` **triple** per line, **1-based** — see §5.2 |
| `.text(limit=None)` / `.bytes(limit=None)` | both are methods, both take a byte limit |
| `.name` `.path` `.relpath` `.size` | where this file is |
| `.bad_lines` | every line this `Doc` could not read, in order |

Two things on that list are newer than the rest and both closed a silent loss.

**Line numbers are 1-based.** They were 0-based — `enumerate()` with no `start` — and this
is the one mechanism whose entire purpose is pointing a person at a bad line, in a world
where every editor, compiler, `sed` and human counts from one. Two authors reported it
independently on the same day.

**`.json()` records a parse failure the way `.jsonl()` records a bad line.**
Before that it did not, and the consequence lived in this page's own favourite idiom,
`src.doc(0).json() or {}`:

```text
intact      .json() -> {...}   bad_lines=[]
empty       .json() -> None    bad_lines=[]
truncated   .json() -> None    bad_lines=[]
```

A truncated file and an empty one were indistinguishable, and the two reads were
`to_dict()`-identical. §1 says this library exists because *a failed search looked exactly
like an empty folder*; a failed **parse** looked exactly like an empty **file**, in the
happy path this page teaches four times over. Now the runtime compares what the `Doc` lost
against the `Unreadable`s you yielded, and reports the shortfall itself — §5.2 and §6.5
are the same check, and they now cover whole documents as well as lines.

`.text()` does its own small version: it decodes with `errors="replace"`, which destroys
bytes, and it records that too. What it cannot do is police a translator that takes the
string and hand-parses it — no closed method list can, and a tracking proxy is defeated by
one `dict()` call. That bypass is real, it is named here rather than papered over, and the
answer to it is `Unreadable`, not a cleverer `Doc`.

### 4.6 `site.row`, not `site.last_segment`

`key=(src.site.row,)` is how an answer says which rollout it is about, and this is the
single most expensive mistake on this page — it cost four independent authors real time.

**`site.last_segment` is only the last path segment.** `site.row` is the segments that
matched a `*`, joined. Where `where` is one segment deep they are equal. Where it is two
deep they are not, and `last_segment` collides:

```python
import tempfile
from pathlib import Path

SCRATCH = tempfile.TemporaryDirectory()

def on_disk(files, name):
    root = Path(SCRATCH.name) / name
    for rel, body in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(body)
    return LocalTree(root)

TWO_DEEP = on_disk({"manifest.json": "{}",
                    "task_a/repeat_0/outcome.json": "{}",
                    "task_a/repeat_1/outcome.json": "{}",
                    "task_b/repeat_0/outcome.json": "{}"}, name="two-deep")
DEEP_CASE = Landmark(name="case", marker="outcome.json", within=RUN, where=("*/*",),
                     expect=Card.AT_LEAST_ONE(because="one directory per attempt"))
found = resolve_landmarks(Index(TWO_DEEP), (RUN, DEEP_CASE))
for s in found.sites["case"]:
    print(f"{s.path:18} last_segment={s.last_segment:10} row={s.row}")
```

```text
task_a/repeat_0    last_segment=repeat_0   row=task_a/repeat_0
task_a/repeat_1    last_segment=repeat_1   row=task_a/repeat_1
task_b/repeat_0    last_segment=repeat_0   row=task_b/repeat_0
```

Two of those three last segments are `repeat_0`. Keyed on `last_segment`,
`task_a/repeat_0` and `task_b/repeat_0` become one row, and the second silently overwrites
or contests the first. **Use `site.row` unless you can say why not.**

`on_disk` above writes a few files to a temporary directory and hands back a `LocalTree` —
useful for exactly this kind of check, and for testing a reader without fixtures.

### 4.7 The input: tying code to declaration

An **input** is one named unit: a translator, the files it reads, and the questions it is
allowed to answer.

```python
OUTCOME = Input(name="outcome", via=outcome,
                reads=(At(CASE, "outcome.json"),),
                answers={"rollout.score":
                         Card.ONE(because="every case that ran was scored")})
print(OUTCOME, "at", OUTCOME.landmark)
```

- `At(CASE, "outcome.json")` — a path relative to a landmark. Rigid, segment by segment.
- `reads=` may hold several patterns, but they must be anchored to **one** landmark, and
  they mean *the same content in more than one place* — usually because the producer moved
  it. Two different kinds of content are two inputs.
- An input **runs once per site of its landmark.** Two case directories means `outcome` is
  called twice, each time seeing only its own directory.
- `answers=` — the questions it may answer, each with a count. Yielding anything else is a
  `Refusal`.
- The count is checked **at every site**, including sites where your file is absent. If a
  case may legitimately have no `outcome.json`, this has to be `Card.ANY`.

### 4.8 The reader

First, the landing check from §4.4, firing. `SCORE` lands in `rollouts.reward`, which
`STORE` has; here is the same reader with a question that lands somewhere it does not:

```python
WRONG = Question(name="rollout.score", grain=Grain.ROLLOUT, merge=Merge.ONE,
                 lands="rollouts.score", means="what this case scored")
try:
    Reader(name="lesson", landmarks=(RUN, CASE), inputs=(OUTCOME,), questions=(WRONG,),
           schema=STORE, counts=Population.COMPLETED,
           rows=(Rows(question=Grain.ROLLOUT, may_create=("outcome",),
                      because="a case directory IS the enumeration"),))
except Refusal as e:
    print(e)
```

```text
Question('rollout.score') lands in 'rollouts.score' and 'rollouts' has no column
'score'. Its columns are ['reward', 'stop_reason']. Either the column is spelt
differently in the store, or the store has no home for this fact — and a fact with no
home is not stored by declaring that it is.
```

With `lands="rollouts.reward"` it constructs, and reads:

```python
LESSON_READER = Reader(
    name="lesson", landmarks=(RUN, CASE), inputs=(OUTCOME,), questions=(SCORE,),
    schema=STORE, counts=Population.COMPLETED,
    rows=(Rows(question=Grain.ROLLOUT, may_create=("outcome",),
               because="a case directory IS the enumeration"),))
out = LESSON_READER.read(LESSON)
print(out.report())
print("ok:", out.ok, " rollouts:", len(out.keys_of("rollout.score")))
```

Two arguments there have no default and cannot be omitted.

**`counts=`** says which rollouts your rows are drawn from: `Population.COMPLETED`,
`ATTEMPTED` or `PREPARED`. It exists because of a measured disaster. One real folder holds
three files:

```text
artifacts/
  evaluator_rollouts.jsonl                        0 lines   finished
  evaluator_rollouts_failures.jsonl              45 lines   crashed
  evaluator_rollouts_materialized_inputs.jsonl  458 lines   prepared to run
```

Four people wrote readers for it independently and produced **0, 45 and 503** rollouts.
Every one passed every other check, because nobody was ever asked which set they meant.

**`rows=`** says which inputs may *create* rows. It has a section of its own (§6.2),
because it is the check most likely to refuse the reader you were about to write.

---

## 5. The library complains, and that is the product

That last run printed **four findings** and two rollouts. A **finding** is data: a record
in a list you can count, filter and store. It is not an exception, and a read with
findings still returns its rows.

There are **eight** kinds, and they do not mean the same thing:

| kind | meaning | fails `ok`? |
|---|---|---|
| `MISSING` | a declared count was not met | yes |
| `UNREAD` | a file no declaration accounts for | yes |
| `COVERAGE` | a total published over an incomplete population; a declared mint that minted nothing | yes |
| `UNREADABLE` | a record a translator was handed and could not read | yes |
| `UNBACKED` | a source described a row that no enumerating source produced | yes |
| `CONTESTED` | two sources disagreed and an `Order` settled it | **no — the feature working** |
| `ORPHAN` | a join matched nothing where nothing was demanded | **no — a join records its misses on purpose** |
| `UNCHECKED` | a declared check that never ran, and what it would have checked (§13) | **no — nothing went wrong; something was not looked at** |

```python
print("kinds:", [k.name for k in Kind])
print("count against ok:", sorted(k.name for k in LOSS))
print("ok:", out.ok, " quiet:", out.quiet)
```

Two flags, and they are not the same question. **`ok` means nothing is unaccounted for** —
gate your pipeline on this. **`quiet` means there is nothing to report at all**, including
the three kinds that do not fail `ok`. `quiet` is a bonus, never a target.

**Do not design your reader to avoid findings.** When every kind counted against one flag,
four separate authors responded by refusing to use joins at all.

### 5.1 `UNREAD`: every file is accounted for

Three of those four findings were these:

```text
[unread] unread @ cases/case_0001/turns.jsonl: no input reads this file and no Ignore
                  accounts for it. An unexamined file stays loud.
[unread] unread @ manifest.json: ...
[unread] unread @ summary.json: ...
```

Nothing is wrong with your scores. But there are three files in that folder that **no
declaration mentions**, and a file nobody looked at is exactly how the version-47 bug hid.
You have two honest answers for each: read it, or say out loud that you are not.

```python
IGNORED = (
    Ignore(RUN, "manifest.json",
           because="benchmark and model already arrive from the control plane"),
    Ignore(RUN, "summary.json",
           because="a rollup of the same scores; the `summary` input of section 6.2 "
                   "reads it, and this Ignore goes away when that input is added"),
    Ignore(RUN, "cases/*/turns.jsonl",
           because="the transcript; the `turns` input of section 5.2 reads it, and this "
                   "Ignore goes away when that input is added"),
)
print([str(i) for i in IGNORED])
```

An `Ignore` is anchored, rigid and reasoned, exactly like a read — because it is a read
that discards. `**` is allowed but must name the directory whose subtree it covers: an
unanchored recursive wildcard is how **871,125 files came to be hidden by four words that
described 8,623 of them**.

What you cannot do is nothing. `because="not looked at yet"` is refused by name.

**A declaration is not a read.** A file counts as read when your translator *looked at*
it — opened its bytes, or asked for its name, path or size. It used to count as read the
moment a pattern matched it, before any translator ran, which meant this guarantee could
be satisfied by a glob: a 27-line reader declaring `At(ROOT, "dir/**")` over a 2,322-file
folder opened not one byte and reported `ok: True, quiet: True, UNREAD: 0`.

"Looked at" is wider than "parsed" on purpose. A translator publishing an inventory of a
deliverables tree — path, name and size, no contents — has read those files in every sense
a person means, and making it pull 254 GB through `bytes()` to prove it would be the
accounting rule making the reader worse.

For files *above* your run landmark there is **`ROOT`**, the landmark that is always there
and never has to be declared:

```python
print(ROOT, ROOT.markers, ROOT.where)
print(Ignore(ROOT, "logs/**", because="scheduler chatter, no evaluation content"))
```

### 5.2 `Unreadable`: saying a record was there and you could not read it

`.jsonl()` yields a **triple**, not a pair, because a file can be cut in half mid-record:

```python
def turns_wrong(src):
    if not src.has(0):
        return
    for line, rec, err in src.doc(0).jsonl():
        if err:
            continue                     # the silent drop
        yield Answer("rollout.events", rec, key=(src.site.row, rec["id"]))
print("translator:", turns_wrong.__name__, "- drops the error branch on purpose")
```

That `continue` turns a file of two records whose second is truncated into a file of one
record, and nothing anywhere says so. The library catches it anyway, because a `Doc`
counts the parse errors it handed out and the runtime compares that count against the
`Unreadable`s you yielded:

```python
EVENTS = Question(name="rollout.events", grain=Grain.EVENT, merge=Merge.UNION,
                  lands="events.kind", means="the transcript this case produced")
TURNS_WRONG = Input(name="turns", via=turns_wrong, reads=(At(CASE, "turns.jsonl"),),
                    answers={"rollout.events":
                             Card.ANY(because="a case with no transcript is legal")})
DROPS = Reader(name="drops", landmarks=(RUN, CASE), inputs=(TURNS_WRONG,),
               questions=(EVENTS,), schema={"events": ["kind"]},
               counts=Population.COMPLETED,
               rows=(Rows(question=Grain.EVENT, may_create=("turns",),
                          because="the transcript file is the enumeration of events"),),
               attach=(Contribute(to="rollout.events", input="turns", order_by="ts",
                                  identity="id", because="the model's own ids and clock"),),
               ignored=IGNORED[:2] + (Ignore(RUN, "cases/*/outcome.json",
                                             because="read by the outcome input"),))
print(DROPS.read(LESSON).report())
```

`rollout.events` is `Merge.UNION` — several files hold parts of one transcript — and every
input answering such a question must declare a `Contribute` saying which column carries
its rows' identity and which one sorts them. §9 is about that; it is here only because the
library will not build the reader without it.

```text
[unreadable] rollout.events @ cases/case_0001 [turns]: 1 record(s) could not be parsed
and the translator reported 0. The rest were dropped without a word — turns.jsonl lines
[1]. ... yield an Unreadable for each one.
```

Write it yourself anyway. Yours can name the record; the library's can only name the line.

```python
def turns(src):
    if not src.has(0):
        return
    doc = src.doc(0)
    for line, rec, err in doc.jsonl():
        if err:
            yield Unreadable(question="rollout.events",
                             where=f"{doc.name} line {line}", detail=err)
            continue
        yield Answer("rollout.events", rec, key=(src.site.row, rec["id"]))
print(Unreadable(question="rollout.events", where="turns.jsonl line 1",
                 detail="unparseable record"))
```

**`Unreadable` is the second thing a translator may yield.** On the 53-folder corpus the
shared JSONL helper the readers go through produces **20 `UNREADABLE` findings** with it,
every one a record cut in half by a 512,000-byte cap in the script that fetched the corpus:
twenty losses that would otherwise be a shorter file. It fills no column and answers
nothing; it reports a loss, with a question, a place and a detail, all three required. Use
it whenever a record is *there* and you could not read it — a truncated line, a field in a
shape you do not recognise, an id you cannot resolve. The alternative is a shorter file
that looks complete.

Note the `src.has(0)` guard: `case_0002` wrote no `turns.jsonl` at all.

Three ways to reach a single file, and they differ only in what they do about absence:

| | matched none | matched several |
|---|---|---|
| `src.doc(i)` | reports the miss; this site yields nothing, the read goes on | `Refusal` |
| `src.first(i)` | `None` | the first, in path order |
| `src.has(i)` | `False` | `True` |

`doc(i)` used to raise `Refusal` on zero matches, and `Refusal` is the one exception this
library raises on purpose — it takes the **whole folder** with it. Two authors lost **9 of
26 held-out folders** to that, on a file a real producer legitimately does not write: a
gym run with zero rollouts writes no aggregate file. Worse, the declaration that would
have caught it properly — `Card.AT_LEAST_ONE` on that read — never ran, because the
translator died first. Zero matches is a fact about bytes, not a bug in a declaration, so
it is now a contained failure: one `MISSING` naming the pattern, this site yields nothing,
every other site and input still reads, and every count this input declared is checked
against the nothing it got.

Matching *several* where the translator asked for one is still a `Refusal`, because that
is a pattern nobody can execute without guessing which file was meant. Use `docs(i)` when
you mean all of them.

If your translator raises instead, the crash is caught, becomes one `UNREADABLE` finding
for that one site, and the rest of the folder still reads. A `Refusal` still propagates —
that is a bug in a declaration, not a fact about bytes.

---

## 6. Ten checks that will refuse the reader you were about to write

These ten became required all at once. Every one refuses, or fails, a reader that an
earlier version of this page would have taught someone to write. Nine of the ten were
already in the library as things you *could* declare, and almost nobody did — the measured
compliance rate with an optional check here was near zero, which is why none of them is
optional any more.

| | the check | what it refuses |
|---|---|---|
| 1 | a count is met by a **value** | an answer carrying `None` satisfying `Card.ONE`, or winning a settle a value was in |
| 2 | every row-grain question declares who may **create** rows | a rollup inventing rollouts |
| 3 | a key space holds **one** value | `keys={"event": (a, b, c)}`, and blank ids |
| 4 | a dropped parse error is a finding | `if err: continue` |
| 5 | a jsonl record that is not an object is a parse error | `isinstance(rec, dict)` guards |
| 6 | ambiguity nothing declared how to settle is a **loss** | two stream rows with one identity |
| 7 | `Relate(to_card=)` has no default | omitting the claim from the other end |
| 8 | `Relate(cardinality=Card.ANY)` where the `frm` side does not mint | a join that resolves nothing, forever |
| 9 | an unmet declared edge count is a **loss** | 40 unresolved edges reported green |
| 10 | `lands=` resolves against the real schema, per **column** | landings naming nothing |

Each has a subsection below. §6.3 sits between two of them and is **not** one of the ten —
`Order` has always been required — because §6.2 and §6.3 are two halves of one example.

### 6.1 A declared count is met by a **value**, not by an answer

`case_0002` wrote `{"score": null}`. Its translator yielded one `Answer`, so an answer
arrived — and that used to satisfy `Card.ONE`.

```python
for f in out.of_kind(Kind.MISSING):
    print(f.line())
```

```text
[missing  ] rollout.score @ case@cases/case_0002 [outcome]:
outcome.rollout.score at ('case_0002',): expected exactly one, found 0
(1 answered with no value) — every case that ran was scored
```

"found 0 (1 answered with no value)" is the whole check in one line: an answer arrived and
it carried nothing, so the count of *values* is zero.

Across 53 corpus folders, **1,325 of 29,460 rows held `None`, and 19 folders reported `ok`
while doing it** — including **558 rollouts whose reader had written, in the `because=`,
"every trial.json carries a reward, 0.0 included".** The author knew zero was a value. The
library counted `None` as one anyway.

Worse, most of those 558 were never null in the file. The reader passed each verdict
through a careful float helper that rejects booleans on purpose — and **523 of the 558
verdicts are booleans**. The helper destroyed 100% of that benchmark's scores and every
read reported `ok`.

Your three honest answers, in order of preference:

1. Read the value properly — `True` is a verdict, convert it to `1.0`.
2. If the producer legitimately writes nothing here, declare `Card.ANY`.
3. If the record was there and you could not read it, yield `Unreadable`.

What is no longer available is yielding `None` and passing.

The same rule applies one step later, where two sources answer one question and an
`Order` picks between them: an order ranks the sources that *stated* a fact, so a `None`
that outranks a real value has not settled anything. That is a `MISSING` too, and §6.3
shows it happening.

### 6.2 Every row-grain question must declare who may create rows

`summary.json` names three cases. Two directories exist. Add a second input that reads the
rollup:

```python
def summary(src):
    for case, score in (src.doc(0).json() or {}).get("cases", {}).items():
        yield Answer("rollout.score", score, key=(case,))

SUMMARY = Input(name="summary", via=summary, reads=(At(RUN, "summary.json"),),
                answers={"rollout.score": Card.AT_LEAST_ONE(
                    because="the rollup holds one row per case that finished")})
OUTCOME_ANY = Input(name="outcome", via=outcome, reads=(At(CASE, "outcome.json"),),
                    answers={"rollout.score": Card.ANY(
                        because="a case that crashed before scoring writes null")})
BOTH = dict(name="lesson2", landmarks=(RUN, CASE), inputs=(OUTCOME_ANY, SUMMARY),
            questions=(SCORE,), schema=STORE, counts=Population.COMPLETED,
            ignored=IGNORED[:1] + IGNORED[2:])
try:
    Reader(**BOTH)
except Refusal as e:
    print(e)
```

Two refusals are waiting in that reader and you get them one at a time. This is the
first:

```text
'rollout.score' is answered by ['outcome', 'summary'] and nothing says which inputs may
CREATE rows at ROLLOUT grain. An Order picks a winner among sources that answered the
same row; it does nothing about rows only one source names, so a summary can invent them,
and a stream's Contribute says who contributes, never which keys they may bring.
Add to rows=:
    Rows(question=Grain.ROLLOUT, may_create=('outcome',), because="...")
  ...which covers every question at that grain, or name this one question to override it.
```

The second, once this one is satisfied, is §6.3's.

An `Order` (§6.3) settles *disagreements about the same row*. It does nothing at all about
rows only one source named. Measured on a real folder: **a manifest listing 267 trials
over 18 directories produced 267 rollouts and passed every check.**

So say who enumerates:

```python
ENUMERATION = Rows(
    question=Grain.ROLLOUT, may_create=("outcome",),
    because="the case directories ARE the enumeration; the rollup is keyed on score "
            "value, so a case that crashed before scoring is absent from it, and a "
            "case it names with no directory did not run")
print(ENUMERATION.question, ENUMERATION.may_create)
```

- `question=Grain.ROLLOUT` covers **every** question at that grain: who enumerates the
  rows at a grain is one decision per reader, not one per fact.
- Name a single question instead where one fact needs a narrower list; a question that
  names itself wins over the grain it sits at.
- Name several inputs where several genuinely enumerate — two files that each list some of
  the cases are peers, not a summary and its subject.
- Everyone not named may **fill rows in, never add them.** A row only a describer named
  becomes an `UNBACKED` finding instead of a rollout.

**A `Merge.UNION` stream is not exempt, and used to be.** The reasoning was that a
stream's contributors are named by a `Contribute` already. `Contribute` names who
contributes; it says nothing about which keys they may bring. Measured: a second
contributor writing keys nobody had seen turned **18 rows over 1 rollout key into 1,893
rows over 158**, with no finding and `ok` True.

**A question only one input answers is not exempt either**, and that is the worse half:
**44 of the 50 real cases are one input enumerating alone**, with nothing to disagree
with it.

Only run grain is exempt, for a checkable reason rather than a judgement: a run-grain
answer carries no row key at all, so there is no row for a describer to invent.

### 6.3 Two sources, one fact: `Order`

*Not one of the ten. `Order` has always been required; it is here because §6.2 and §6.3
are the two halves of one example.*

```python
try:
    Reader(**BOTH, rows=(ENUMERATION,))
except Refusal as e:
    print(e)
```

```text
Reader(lesson2): 'rollout.score' can be answered by ['outcome', 'summary'] and no order
is declared. Declaration order would decide it silently. Add:
    Order(question='rollout.score', order=('outcome', 'summary'), keep_loser=True,
          because="...")
```

It prints the line to paste; you supply the reason.

```python
RANK = Order(question="rollout.score", order=("outcome", "summary"), keep_loser=True,
             because="the rollup is keyed on score value, so a case that crashed before "
                     "scoring is simply absent from it; preferring it drops the row, "
                     "not just the number")
LESSON2 = Reader(**BOTH, rows=(ENUMERATION,), orders=(RANK,))
out2 = LESSON2.read(LESSON)
print(out2.report())
print("ok:", out2.ok, " rollouts:", len(out2.keys_of("rollout.score")))
```

```text
[missing  ] rollout.score @ ('case_0002',) [outcome]: outcome won this settle carrying no
            value, and summary=0.0. That value is gone and no declaration chose to lose
            it ... If outcome means there is genuinely nothing here, say it with
            Unreadable or do not answer at all — then the source that did answer is the
            answer. If it does not mean that, the order is ranking the wrong source first
[unbacked ] rollout.score @ ('case_0003',) [summary]: 'summary' describes this row and no
            input in ['outcome'] — the inputs that may create rows at ROLLOUT grain —
            produced it, so it is not a row — ...
```

Three cases named, two directories, **two rollouts**. `case_0003` is an `UNBACKED`
finding, not a row.

- `order=` — first name wins. It must be **total**: every source that can answer, and no
  source that cannot. Sources are names, not objects, because a source can be an input
  (`outcome`), an input and its decoder (`turns/atif-1.6`, §8), or a fold
  (`fold:rollout.cost`, §7).
- `keep_loser=True` — keep the beaten value, reachable as `out.losers`. You almost always
  want this. The disagreement is data, and `CONTESTED` does not fail the read.
- One order per question. An `Order` on a `Merge.UNION` question is refused.

**Read that first line again.** Your order put `outcome` above `summary`, and for
`case_0002` `outcome` answered `None` while `summary` had a real `0.0`. That is not a
disagreement your order settled: an order ranks the sources that **stated** a fact, and
`None` is the absence of one. So it is a `MISSING` — a loss, `ok` False — and not a
`CONTESTED`, which would mean the ranking did its job and would leave the read green.

The stored value is still `None`, because the library will not guess which of two things
you meant. Two repairs, and you pick:

- `outcome` genuinely has no score for this case → have the translator yield
  `Unreadable`, or yield nothing at all. Then `summary` is the only answer, and the
  `0.0` is stored with no finding.
- `outcome` should not outrank a source that has the number → change the order.

Until you do one of them, the read is red. That is the point: nothing here quietly picks
for you, and the finding names both lines you could write.

**What `Order` cannot do:** it ranks *sources*, not documents. If one pattern matches two
files — the same manifest written twice, 23 hours apart — there is no way to say "take the
newer one". Make the question `Merge.UNION` so both land and the decision is visibly not
taken.

### 6.4 A key space holds one value

A **key space** is a declared name for a family of ids (§10). A value in one names **one**
thing:

```python
try:
    Answer("rollout.events", {"id": 1}, key=("c1",), keys={"event": ("e1", "e2", "e3")})
except Refusal as e:
    print(e)
```

A tuple is hashable, so nothing structural objects — but every consumer looks the value up
whole, so it matches nothing, ever. Measured on one real folder: **1,243 edges, 0
resolved, 1,243 orphans, `ok` True.** One answer names one thing; three references are
three answers carrying the same row key.

A blank string is refused for the same reason from the other side:

```python
try:
    Answer("rollout.events", {"id": 1}, key=("c1",), keys={"event": ""})
except Refusal as e:
    print(e)
```

A blank id joins every other row that also has a blank one. A key the producer did not
write is a key this answer does not carry — leave the space out of `keys`.

### 6.5 A dropped parse error is a finding

Covered in §5.2. The check is that the `Doc` counts what it handed out, so no helper can
undo it: **six of seven shipped readers dropped quarantined lines**, through one shared
helper whose own docstring said so. Fixing the helper is one edit; the edit after it puts
the `continue` back.

A line that parses to something that is not an object counts too — a list, a number, a
bare string:

```python
POISON = on_disk({"t.jsonl": '{"id": 1}\n["not an object"]\n{"id": 2\n'}, name="poison")
doc = Doc(Index(POISON), Index(POISON).files[0], Site("run", "", ""))
for line, rec, err in doc.jsonl():
    print(line, rec, "|", err)
print("bad_lines:", doc.bad_lines)
```

A translator asking for records gets `None` for that middle line, and the `isinstance`
check that drops it is exactly as silent as ignoring the error.

One honest note: across 53 corpus folders, the number of genuinely unparseable producer
lines is **zero**. Every bad line found was an artefact of the script that downloaded the
corpus. This check is a latent hazard with no measured loss here — it is switched on
because a poisoned line should be loud, not because it has already cost anything.

### 6.6 Ambiguity nothing declared how to settle is a loss

Two inputs contribute to one `Merge.UNION` stream and both claim identity `e1`, with
different values:

```python
def turns_a(src):
    yield Answer("rollout.events", {"id": "e1", "ts": 1, "kind": "say"},
                 key=("c1",), input_key="e1")
def turns_b(src):
    yield Answer("rollout.events", {"id": "e1", "ts": 2, "kind": "call"},
                 key=("c1",), input_key="e1")
print("both claim identity e1, with different `kind` values")
```

```python
FLAT = on_disk({"manifest.json": "{}"}, name="one-file")
STREAM = Question(name="rollout.events", grain=Grain.ROLLOUT, merge=Merge.UNION,
                  lands="events.kind", means="the transcript this case produced")
A = Input(name="turns", via=turns_a, reads=(At(RUN, "manifest.json"),),
          answers={"rollout.events": Card.ANY(because="a case may write no turn")})
B = Input(name="tools", via=turns_b, reads=(At(RUN, "manifest.json"),),
          answers={"rollout.events": Card.ANY(because="a case may call no tool")})
COLLIDE = Reader(
    name="collide", landmarks=(RUN,), inputs=(A, B), questions=(STREAM,),
    schema={"events": ["kind"]}, counts=Population.COMPLETED,
    rows=(Rows(question=Grain.ROLLOUT, may_create=("turns", "tools"),
               because="both components write turns of the same conversation"),),
    attach=(Contribute(to="rollout.events", input="turns", order_by="ts", identity="id",
                       because="the model's own ids and clock"),
            Contribute(to="rollout.events", input="tools", order_by="ts", identity="id",
                       because="a different component, with its own ids")))
print(COLLIDE.read(FLAT).report())
```

That used to be a `CONTESTED` — the kind that does not fail a read. It is wrong, and the
new message says why: `Merge.UNION` **refuses** an `Order`, so nothing chose between the
two values. Sort position did, and the loser's value is gone. There were **135,585 of
these in one corpus, every one green.** It is now `MISSING`.

Your two fixes are in the message: give the two rows distinct identities, or make the
question `Merge.ONE` and rank the sources.

### 6.7–6.9 The three on `Relate`

`Relate` records an edge between two rows (§10). Three of the ten live here.

**6.7 — `to_card` has no default.** Omit it and you get a sentence naming the line to
write, like every other check here:

```python
try:
    Relate(frm="rollout.artifacts", to="rollout.events", on=Join("event"),
           cardinality=Card.ONE, because="an artifact is written by one event")
except Refusal as e:
    print(str(e)[:96], "...")
```

`cardinality` is how many `to` rows each `frm` row must reach; `to_card` is the same
question from the other end. `to_card=None` says you are not making the second claim —
right for a leaf of a DAG, wrong for a sub-agent nobody spawned. Writing `None` is a
decision somebody made; omitting it would be one nobody made.

**6.8 — `Card.ANY` is refused where the `frm` side does not mint the key space.** See §10 for
what minting is; the rule is that if your row carries an id it read out of somebody else's
bytes, the producer is asserting the thing exists, so `Card.ANY` is not a weaker claim —
it is no claim:

```python
try:
    Relate(frm="rollout.artifacts", to="rollout.events", on=Join("event"),
           cardinality=Card.ANY, to_card=None,
           because="a deliverable in the file tree may have no event")
    print("the Relate constructs on its own — the refusal happens at Reader()")
except Refusal as e:
    print(e)
```

That one is checked on the `Reader`, because only the reader knows who mints what. With
the ids deliberately put in the wrong family: `Card.ONE` gave `ok` False, 0 of 1,501
resolved, 1,501 findings; `Card.ANY` gave `ok` True, 0 of 1,501 resolved, **and nothing at
all**.

**6.9 — an unmet declared cardinality is a `MISSING`, not an `ORPHAN`.** `ORPHAN` does not fail
a read, and **40 unmet edges left `ok` True** while resolving nothing. The severity now
follows what you declared: `Card.ANY` leaves an orphan, anything stronger is a loss. §10
shows it firing.

### 6.10 `lands=` resolves against the real schema, per column

§4.4. It is the library's one machine-checkable declaration, and until it was checked most
landings in the shipped readers named nothing. **Per column is the honest unit**, because
a landing can pass on its table existing while the column it needs does not.

That unit matters twice, and the second time is easy to miss. A landing may name a whole
table — `lands="metrics"` — when one answer is a whole row of it. Registration can only
check that `metrics` exists, because *which* columns the question fills is the mapping
your translator yields, and no mapping exists until a file is read. So the check runs
again during the read: every key of every row is resolved against that table's columns,
and a key the table has no column for is a `MISSING`. Measured on the readers shipped
here, two questions land in a whole table and **between them they fill not one column of
it**, over nine folders that were previously silent. §12 uses the whole-table form
correctly, and says how to tell.

---

## 7. Lifting a number up a grain: `Fold`

The run's total cost is the sum of the cases' costs.

```python
COST = Question(name="rollout.cost", grain=Grain.ROLLOUT, merge=Merge.ONE,
                lands="rollouts.cost", means="what this case cost")
RUN_COST = Question(name="run.cost", grain=Grain.RUN, merge=Merge.ONE,
                    lands="runs.cost", means="what the whole run cost")
TOTAL = Fold(into="run.cost", over="rollout.cost", op=Op.SUM,
             expect=Card.NONE(because="every case that ran reports a cost"),
             because="cost is extensive in cases: two cases cost what each of them cost")
print(TOTAL.into, "<-", TOTAL.op.name, "over", TOTAL.over)
print("stamped source:", TOTAL.source, " is an aggregate:", TOTAL.aggregate)
```

- `op` is a closed vocabulary — `SUM`, `MEAN`, `MIN`, `MAX`, `COUNT`, `DISTINCT` — not a
  callable you pass in. `SUM` versus `MEAN` is a real decision, and a lambda would hide it
  from the diff.
- A fold lifts **exactly one rung**: part/call → event → rollout → run.
- `expect=` is **not** a second spelling of the low-grain count. It counts the subjects
  that did **not** contribute, and answers the one question that is new at a grain
  boundary: may this total be published when the population is incomplete? Only
  `Card.NONE` (no) and `Card.ANY` (yes, and name the shortfall) mean anything there.

```python
try:
    Fold(into="run.cost", over="rollout.cost", op=Op.SUM,
         expect=Card.ONE(because="one total per run"),
         because="cost is extensive in cases")
except Refusal as e:
    print(e)
```

Either way the shortfall is reported as a `COVERAGE` finding, which **fails `ok`**.
`Card.ANY` does not silence it; nothing does — but read the next paragraph before you
trust that sentence, because for one very common shape there was nothing to silence.

**A subject is a row some input named at the low grain.** So if the folded question is the
**only** question at its grain, every subject is by definition one that answered it, the
shortfall is zero whatever the folder holds, and `expect=Card.NONE` reports success by
construction. Demonstrated: 8 of 59 attempts skipped, that `expect` declared in writing,
**0 `COVERAGE` findings, and a benchmark score 16% overstated.** A check that cannot fail
is what this library elsewhere calls a worse foundation than no declaration, so it is now
refused where it is written:

```python
try:
    ONLY = Question(name="solo.cost", grain=Grain.ROLLOUT, merge=Merge.ONE,
                    lands="rollouts.cost", means="the only thing said about a rollout")
    TOP = Question(name="solo.total", grain=Grain.RUN, merge=Merge.ONE,
                   lands="runs.cost", means="their total")
    Reader(name="solo", counts=Population.COMPLETED, landmarks=(RUN, CASE),
           inputs=(Input(name="only", via=lambda src: iter(()),
                         reads=(At(CASE, "outcome.json"),),
                         answers={"solo.cost": Card.ONE(because="every case costs")}),),
           questions=(ONLY, TOP),
           rows=(Rows(question=Grain.ROLLOUT, may_create=("only",),
                      because="it is the enumeration"),),
           folds=(Fold(into="solo.total", over="solo.cost", op=Op.SUM,
                       expect=Card.NONE(because="no total over a partial population"),
                       because="cost is extensive in cases"),))
except Refusal as e:
    print(str(e)[:200])
```

The way out is one line in the enumerating translator: answer a **second** question at
that grain — a rollout id is enough. The rows it names and the folded question does not
are then the shortfall, and `expect` has something to count. If you genuinely cannot know
the denominator, `Card.ANY` is legal there and says so in writing.

The folded value is stamped with the source name `fold:rollout.cost`, which is what an
`Order` ranks it by if something else also answers `run.cost`. Unless the fold is an
aggregate, in which case that collision is refused outright:

```python
print("aggregate ops:", sorted(o.name for o in AGGREGATE_OPS))
```

A `SUM` over a population and a number stated in a document are two different facts
sharing a name. An `Order` between them does not pick a winner — it deletes the other
question. Give the stated one its own question.

---

## 8. One file, two record formats: `Decoders`

A producer changed its record format mid-file. One real trace file held **812 records in
one version and 88 in another, in the same file**. So "which format is this file?" is the
wrong question; the right one is "which format is this *record*?"

```python
V16 = Decoder("atif-1.6", because="records carrying `v: 1.6`, with id/ts/role")
V15 = Decoder("atif-1.5", because="records carrying `v: 1.5`, with event_id/t/actor")
CANDIDATES = Decoders(candidates=(V16, V15),
                      because="one real trace file held 812 records of v1.5 and 88 of v1.6")
print(CANDIDATES.names)
```

`candidates` is a **set of what may be used, not a ranking**: where several could read one
record, that is an `Order` over the stamped source names, not a race won by tuple
position. Your translator stamps each answer with the decoder that read it:

```python
def versioned(src):
    for line, rec, err in src.doc(0).jsonl():
        if err:
            yield Unreadable("rollout.events", f"line {line}", err)
            continue
        via = V16.name if rec.get("v") == "1.6" else V15.name
        yield Answer("rollout.events", rec, key=(src.site.row,),
                     input_key=str(rec.get("id") or rec.get("event_id")), via=via)
print("stamped sources become:", [f"turns/{n}" for n in CANDIDATES.names])
```

Two things to know before you use this:

- **Declaring `decoders` splits one input into several sources.** `turns` becomes
  `turns/atif-1.6` and `turns/atif-1.5`. If the question is `Merge.ONE`, you must now
  write an `Order` between them, even though only one can apply to a given record.
- If you declare decoders you must stamp **every** answer with `via=`; if you do not
  declare them you may stamp none. Both halves are refused.
- A one-candidate `Decoders` needs no order and cannot be given one — an `Order` of one
  name is refused.

---

## 9. Streams: `Merge.UNION` and `Contribute`

A conversation is usually split across files written by different components:

```text
cases/case_0001/
  turns.jsonl       the model's own turns, with its ids and its clock   -> input `turns`
  tools.jsonl       the tool calls, written by a different component    -> input `tools`
```

Both belong in one event stream. That is what `Merge.UNION` is for, and every input
answering such a question **must** declare a `Contribute`:

```python
print(Contribute(to="rollout.events", input="turns", order_by="ts", identity="id",
                 because="the model's own turns, with the producer's ids and clock"))
```

- `order_by` — the column **in this input's own rows** the merged stream is sorted on.
- `identity` — the column that says two rows are the same row, so duplicates collapse.
  Collapsing two *different* values is the §6.6 finding.
- The value of a stream answer must be a **mapping**, so those two columns can be
  addressed. A string is refused.

Rows are named `(input, input_key)` — the input's own id, counted inside its own output —
never by position in the merged stream. So adding a third contributor later renumbers
nothing. `row.position` is assigned last, for display.

---

## 10. Joining: `Key`, `Join`, and the three attachments

A server log has the latency of each model call. The transcript has the calls. You want
the latency on the call's row.

A **key space** is a declared name for a family of ids. It must be declared, because two
id families share one English word all the time: a server log writes a *completion* id
(`chatcmpl-…`) and a transcript column holds a *tool-call* id (`call_…`), both called "the
call id". Joining them passes every structural check and returns zero matches forever.

The input where an id **enters the reader** declares it:

```python
MINT = Key("chatcmpl", because="response.id — the capture file is where that id enters")
OPTIONAL = Key("task", optional=True,
               because="gdpval folders carry a task uuid; the other benchmarks do not, "
                       "and a rollout without one simply does not join")
print(MINT.space, MINT.optional, "|", OPTIONAL.space, OPTIONAL.optional)
```

Three rules, all checked: every key space a join names has a minter; **exactly one** input
mints it; and an input that declares a mint and emits no key for it is a `COVERAGE`
finding. `optional=True` turns off the last one — one reader covers seventeen folders and
two of them carry a task uuid; without the flag, declaring it turns the other fifteen red,
so people delete the declaration and write the join as data nothing checks.

**A join names a key space, never an input.** That is the whole mechanism, and it is why
input execution order cannot matter: nothing points at an input, so nothing implies an
order between them.

The two attachments:

| | what it does |
|---|---|
| `Contribute` | adds rows to a stream, naming nobody else's key |
| `Relate` | records an edge between two rows, matched on a key space |

### `Relate`, end to end

```python
def arts(src):
    yield Answer("rollout.artifacts", {"path": "report.pdf"}, key=("c1",),
                 input_key="report.pdf", keys={"event": "e9"})
def trace(src):
    yield Answer("rollout.events", {"id": "e1", "ts": 1}, key=("c1", "e1"),
                 input_key="e1", keys={"event": "e1"})
ARTS = Input(name="arts", via=arts, reads=(At(RUN, "manifest.json"),),
             answers={"rollout.artifacts": Card.ANY(because="an agent may produce nothing")})
TRACE = Input(name="trace", via=trace, reads=(At(RUN, "manifest.json"),),
              answers={"rollout.events": Card.ANY(because="a case may write no turn")},
              mints=(Key("event", because="the transcript's own record ids"),))
print(TRACE.mints)
```

```python
ARTIFACTS = Question(name="rollout.artifacts", grain=Grain.ROLLOUT, merge=Merge.UNION,
                     lands="artifacts.path", means="files the agent produced")
EVENTS2 = Question(name="rollout.events", grain=Grain.EVENT, merge=Merge.UNION,
                   lands="events.kind", means="the transcript this case produced")
EDGE = dict(name="edge", landmarks=(RUN,), inputs=(ARTS, TRACE),
            questions=(ARTIFACTS, EVENTS2), counts=Population.COMPLETED,
            schema={"artifacts": ["path"], "events": ["kind"]},
            rows=(Rows(question=Grain.ROLLOUT, may_create=("arts",),
                       because="the deliverables directory is the enumeration"),
                  Rows(question=Grain.EVENT, may_create=("trace",),
                       because="the transcript is the enumeration of events")))
ATTACH = (Contribute(to="rollout.artifacts", input="arts", order_by="path",
                     identity="path", because="a deliverable is named by its path"),
          Contribute(to="rollout.events", input="trace", order_by="ts", identity="id",
                     because="the model's own ids and clock"))
print("attachments:", [type(a).__name__ for a in ATTACH])
```

Now the §6.8 refusal, live:

```python
try:
    Reader(**EDGE, attach=ATTACH + (
        Relate(frm="rollout.artifacts", to="rollout.events", on=Join("event"),
               cardinality=Card.ANY, to_card=None,
               because="a deliverable found in the file tree may have no event"),))
except Refusal as e:
    print(e)
```

And with the count the library asks for, the unresolved edge is a `MISSING` (§6.9), the
row keeps a note, and the edge is recorded with no target rather than dropped:

```python
EDGES = Reader(**EDGE, attach=ATTACH + (
    Relate(frm="rollout.artifacts", to="rollout.events", on=Join("event"),
           cardinality=Card.ONE, to_card=None,
           because="an agent deliverable is written by exactly one event"),))
edged = EDGES.read(FLAT)
print(edged.report())
for e in edged.edges:
    print("edge:", e.frm_name, "->", e.to_name, "|", e.reason)
```

`Relate` cannot join a question to itself. If both ends of your edge live in one stream,
make the edge its own question with its own landing.

### One rollout named two ways: `same_rollout=True`

An edge names two rows. Sometimes the two rows are one rollout that two files key
differently, and an edge is the wrong answer. gdpval is the measured case: the stream keys
a rollout `t5-r0` (`_ng_task_index`, `_ng_rollout_index`), the verdicts sidecar keys it
the same way, and the deliverables directory keys it `task_<uuid>/repeat_0`. Only the
stream record states both — in its `deliverables_dir` — so no key exists that all three
files can compute, and without the declaration every task was stored twice (440 rollouts
for a run of 220).

```python
Relate(frm="rollout.n_model_calls", to="rollout.task", on=Join("deliv_dir"),
       cardinality=Card.ONE, to_card=None, same_rollout=True,
       because="a stream record names its deliverables directory")
```

- Before anything settles, every answer under the `frm` rollout's key moves to the `to`
  rollout's key. `Order`, `Rows` and `Contribute` then treat them as one row, which is
  what makes a priority between the two files possible at all.
- Only where the join resolves to exactly one `to` rollout. A key naming two is a
  `CONTESTED` finding and moves nothing; a key naming none stays where it is, and the
  join's own cardinality reports it.
- Both ends must be ROLLOUT grain. Rows below a rollout move with it.
- The `to` end keeps its key, so point the edge at the enumeration.

---

## 11. When the file name is not enough

A file name says *which file*. It does not say *which version of it*, and producers
version their own files:

```text
artifacts/manifest.json
  { "schema_version": "artifacts-v1", "generated_at": … }
```

The day that says `v2`, the name is unchanged, your landmark still resolves, and your
reader reads a layout it has never seen. So a marker may also be required to *say*
something, in the first 256 KB of the file it names:

```python
VERSIONED = Landmark(
    name="run_v1", where=("", "artifacts"),
    marker=Says("manifest.json", contains="artifacts-v1",
                because="the manifest names its own schema; v1 is the only one this "
                        "reader was written against"),
    expect=Card.ONE(because="one manifest per run"))
print(VERSIONED.says, VERSIONED.markers)
```

A literal substring only. No regex (that is code), no JSON path (it needs the window to
parse, and a truncated window may not), and **no negative form** — there is no
`contains_not=` parameter and there will not be one. Absence is not evidence: a missing
literal, a renamed field and a truncated window look identical.

The literal must be at least three characters, for the same reason a one-character marker
is useless:

```python
try:
    Says("manifest.json", contains="v2", because="the version this reader was written for")
except Refusal as e:
    print(e)
```

An empty or unreadable marker is a finding, not a quiet non-match — a failed look that is
indistinguishable from an absent file is the defect this whole design exists to remove.
256 KB is fixed, not a knob: across every marker file in the corpus the discriminating
field sits in the first 17,739 bytes, so the bound is fourteen times the worst real case
and nobody has to tune it.

Where a producer renamed its marker between versions, `AnyOf` is the whole of *or*:

```python
RENAMED = Landmark(name="run_either", marker=AnyOf("lock.json", "manifest.json"),
                   where=("",), expect=Card.ONE(because="v3 renamed the lock file"))
print(RENAMED.markers)
```

`AnyOf` is flat on purpose — nesting it would build an expression whose failure can only
report that it was false, instead of naming the spelling that was missing.

| | how to say it | why |
|---|---|---|
| **and** | two landmarks, each named | the failure says *which* one failed, in your words |
| **or** | `marker=AnyOf("a.json", "b.json")` | a producer renamed its marker |
| **not** | you cannot | absence is not evidence |

Telling "A and B" from "A alone" needs no *not*. The reader that wants both declares two
landmarks and does not resolve where only one is there. The overlap is settled by a
declared order between readers, and the rule is **the reader that needs less goes last**:

```python
SMALL = Reader(name="small", landmarks=(RUN,), inputs=(ARTS,), questions=(ARTIFACTS,),
               counts=Population.PREPARED, schema={"artifacts": ["path"]},
               rows=(Rows(question=Grain.ROLLOUT, may_create=("arts",),
                          because="the deliverables directory is the enumeration"),),
               attach=(ATTACH[0],))
PICK = Dispatch((EDGES, SMALL),
                because="`edge` needs a transcript as well as a manifest, and `small` "
                        "needs only the manifest, so the one that needs less goes last")
print(PICK.pick(FLAT))
```

`Dispatch` needs at least two readers — an order of one settles nothing. Reading the less
specific reader alone says nothing about its being meant to lose; that lives here, in the
`because=`, naming what the earlier reader excludes.

---

## 12. A reader for a real folder

Everything so far was a sandbox. This is a folder from the corpus checkout (see the top of
this page), written by `nemo-skills` through the `nemo-evaluator` wrapper:

```text
corpus2/train/skills@0.4__nemo-skills__mmlu__01964a27/
  artifacts/
    run_config.yml                     what was run, and against what model
    metadata.yaml                      launcher bookkeeping: cluster, mounts, sbatch
    results.yml                        the command line AND the published metric tree
    eval_factory_metrics.json          request statistics
    eval-results/mmlu/
      metrics.json                     published aggregates, 18 of them
      output.jsonl                     30 lines, one scored sample each
```

Six files. Every one of them gets an input or an `Ignore`.

**Two landmarks.** The run is found by `run_config.yml`, which the wrapper writes for
`nemo-skills` and `nemo-gym` alike; `eval-results/<benchmark>/` is what separates the two,
and it is also the enumeration of what this run scored.

```python
NS_RUN = Landmark(
    name="ns_run", marker="run_config.yml", where=("", "artifacts"),
    expect=Card.ONE(because="the wrapper writes exactly one run_config.yml per evaluation"))
NS_TASK = Landmark(
    name="ns_task", marker="metrics.json", within=NS_RUN, where=("eval-results/*",),
    expect=Card.AT_LEAST_ONE(
        because="eval-results/<benchmark>/ IS the enumeration of what this run scored, "
                "and its absence is what tells nemo-gym apart from nemo-skills"))
print(NS_RUN, NS_TASK)
```

Note `where=("", "artifacts")`: the same producer writes the config at the top of the
folder in some versions and under `artifacts/` in others. Two places, one landmark — that
is what several `where` entries are for.

**Three translators.** Plain functions: they open no path, name no column and pick no
winner between two files.

```python
import yaml

def ns_config(src):
    body = yaml.safe_load(src.doc(0).text()) or {}
    yield Answer("run.benchmark", body["config"]["params"]["task"])
    yield Answer("run.model", body["target"]["api_endpoint"]["model_id"])

def ns_totals(src):
    body = src.doc(0).json() or {}
    for subject, aggregates in body.items():
        for name, values in aggregates.items():
            yield Answer("run.metrics",
                         {"subject": subject, "aggregate": name,
                          "n": values.get("num_entries"),
                          "percent": values.get("symbolic_correct")},
                         input_key=f"{subject}/{name}")
print("translators:", ns_config.__name__, ns_totals.__name__)
```

The third one is where §6.1 is earned. `symbolic_correct` in this file is a **boolean on
every one of the 30 records**, and the float helper that a shipped reader used rejects
booleans on purpose — turning every verdict into `None`, which the library then counted as
an answer. Convert it:

```python
def ns_outputs_v1(src):
    for doc in src.docs(0):
        for line, rec, err in doc.jsonl():
            key = (f"{src.site.row}/{doc.name}/{line}",)
            if err:
                yield Unreadable(question="rollout.score",
                                 where=f"{doc.name} line {line}", detail=err)
                continue
            verdict = rec.get("symbolic_correct")
            yield Answer("rollout.score", 1.0 if verdict else 0.0, key=key)
            yield Answer("rollout.stop_reason", rec.get("finish_reason"), key=key)
print("translator:", ns_outputs_v1.__name__)
```

It is called `_v1` because it is wrong, and §12.1 is about finding out how.

Note `src.docs(0)`, plural. The folder this section names holds **one** `output.jsonl`;
its three siblings from the same producer hold **8, 8 and 64**. `src.doc(0)` asks for
exactly one and reports a miss on any other number — which is right where the declaration
means one file, and is not what this declaration means. The earlier version of this page
used `doc(0)` here, and what it taught did not travel three folders sideways.

The row key is `(f"{src.site.row}/{doc.name}/{line}",)`. `site.row` is `mmlu`, and no
record in this file carries an id of its own — **`problem_idx` is absent from all 30** —
so the file name and the line number inside the named benchmark are the address. The file
name is in there because 64 files each numbering their own lines from 1 collide otherwise.
One component, because `rollout.score` is at rollout grain.

**Three inputs.**

```python
NS_CONFIG = Input(
    name="config", via=ns_config, reads=(At(NS_RUN, "run_config.yml"),),
    answers={"run.benchmark": Card.ONE(because="config.params.task is the benchmark run"),
             "run.model": Card.ONE(because="target.api_endpoint.model_id is the model")})
NS_TOTALS = Input(
    name="totals", via=ns_totals, reads=(At(NS_TASK, "metrics.json"),),
    answers={"run.metrics": Card.AT_LEAST_ONE(
        because="the task directory exists because metrics were written into it")})
NS_OUTPUTS = Input(
    name="outputs", via=ns_outputs_v1, reads=(At(NS_TASK, "output*.jsonl"),),
    answers={"rollout.score": Card.ONE(because="every line carries a verdict"),
             "rollout.stop_reason": Card.ONE(because="finish_reason is on every record")})
print(NS_CONFIG, NS_TOTALS, NS_OUTPUTS)
```

**Five questions**, and the schema they must resolve against.

```python
NS_QUESTIONS = (
    Question(name="run.benchmark", grain=Grain.RUN, merge=Merge.ONE,
             lands="runs.benchmark", means="what this run evaluated"),
    Question(name="run.model", grain=Grain.RUN, merge=Merge.ONE,
             lands="runs.model", means="the model under test"),
    Question(name="run.metrics", grain=Grain.RUN, merge=Merge.UNION,
             lands="metrics", means="every published aggregate, one row each"),
    Question(name="rollout.score", grain=Grain.ROLLOUT, merge=Merge.ONE,
             lands="rollouts.reward", means="1.0 if this answer was judged correct"),
    Question(name="rollout.stop_reason", grain=Grain.ROLLOUT, merge=Merge.ONE,
             lands="rollouts.stop_reason", means="why generation stopped"))
NS_SCHEMA = {"runs": ["benchmark", "model"],
             "rollouts": ["reward", "stop_reason"],
             "metrics": ["subject", "aggregate", "n", "percent"]}
print(len(NS_QUESTIONS), sorted(NS_SCHEMA))
```

`run.metrics` lands in a whole table rather than a column, because one answer is a whole
row there. A landing may be `table` or `table.column`, and the two forms may not overlap:
one question landing in `metrics` and another in `metrics.percent` is the same landing
said twice, and is refused.

The whole-table form is the one place a landing is checked **during the read** as well as
at registration: every key of every row — `subject`, `aggregate`, `n`, `percent` — is
resolved against the four columns of `metrics`. Add a fifth key to that dict, or misspell
one, and you get a `MISSING` naming the key and listing the columns the table does have.
That is what keeps `lands="metrics"` from being a way to switch the schema check off, and
it is why the schema above names all four columns rather than just the table.

**The reader.**

```python
NS_READER = Reader(
    name="nemo-skills-lesson", counts=Population.COMPLETED,
    landmarks=(NS_RUN, NS_TASK), inputs=(NS_CONFIG, NS_TOTALS, NS_OUTPUTS),
    questions=NS_QUESTIONS, schema=NS_SCHEMA,
    rows=(Rows(question=Grain.ROLLOUT, may_create=("outputs",),
               because="the output*.jsonl lines ARE the enumeration of scored samples"),),
    attach=(Contribute(to="run.metrics", input="totals", order_by="subject",
                       identity=("subject", "aggregate"),
                       because="one metrics.json per benchmark directory, each row a "
                               "different published aggregate — so the subject alone is "
                               "not an identity, and 18 rows would dedupe down to 1"),),
    ignored=(
        Ignore(NS_RUN, "metadata.yaml", because="launcher bookkeeping: cluster, mounts, "
                                                "sbatch flags. No evaluation content"),
        Ignore(NS_RUN, "results.yml", because="it holds the resolved command line AND a "
                                              "`results:` tree of published metrics — "
                                              "8,038 numeric leaves across the 15 such "
                                              "files in corpus2/train. This lesson reads "
                                              "its aggregates from each benchmark's own "
                                              "metrics.json instead, and a real reader "
                                              "should prefer this file: it is the one "
                                              "both nemo-* producers write identically"),
        Ignore(NS_RUN, "eval_factory_metrics.json",
               because="HTTP request statistics: token counts, latencies, status codes")))
print(NS_READER)
```

`counts=Population.COMPLETED` because a line in `output.jsonl` is written when a sample
finished. Samples that crashed leave no line here, so these rows are the completed set and
nothing in this folder says how many there were meant to be.

**Run it.**

```python
FOLDER = "corpus2/train/skills@0.4__nemo-skills__mmlu__01964a27"
result = NS_READER.read(FOLDER)
print(result.report())
print("ok:", result.ok, " quiet:", result.quiet)
print("benchmark:", result.value("run.benchmark"),
      " model:", result.value("run.model"))
print("rollouts:", len(result.keys_of("rollout.score")),
      " metric rows:", len(result.stream("run.metrics")))
```

```text
nemo-skills-lesson @ skills@0.4__nemo-skills__mmlu__01964a27: claimed, counting completed rollouts
ok: True  quiet: True
benchmark: mmlu  model: qwen35-9B-wa-max
rollouts: 30  metric rows: 18
```

Six files, three read, three ignored, **30 rollouts, 18 metric rows, no findings.**

And that is exactly where a tutorial normally stops, which is why this one does not. A
worked example that only ever succeeds teaches nothing about a library whose entire
subject is failure.

### 12.1 Run it on the folder next door

`nemo-skills` wrote three more folders in `corpus2/train`, by the same producer, same
version. Point the same reader at one of them:

```python
IFBENCH = "corpus2/train/skills@0.4__nemo-skills__ifbench__1518415e"
bad = NS_READER.read(IFBENCH)
scores = [bad.value("rollout.score", k) for k in bad.keys_of("rollout.score")]
print("ok:", bad.ok, " quiet:", bad.quiet, " findings:", len(bad.findings))
print("rollouts:", len(scores), " zeroes:", scores.count(0.0), " ones:", scores.count(1.0))
```

```text
ok: True  quiet: True  findings: 0
rollouts: 31  zeroes: 31  ones: 0
```

**Thirty-one rollouts, every one scored zero, and the library says there is nothing to
report.** ifbench has no `symbolic_correct` field at all — it scores with `loose_eval` and
an `instruction_id_list`. `rec.get("symbolic_correct")` returns `None`, and

```text
1.0 if verdict else 0.0
```

turns that `None` into the number `0.0`. The null rule from §6.1 never fires, because by
the time the answer is made there is no null left to see: **a helper expressing "I do not
know" as a number is the one defect this page spends §6.1 warning you about, and §12 was
committing it.** It is the same shape as the float helper that rejected booleans and
destroyed 100% of another benchmark's scores.

Nothing else in the vocabulary can save you here. `Card.ONE` is met — an answer arrived
carrying a value. `Rows` is satisfied — the enumerating input made every key. The landing
resolves. The schema check passes. Every declaration in that reader is correct, and the
result is a benchmark scored zero under a green light. **The declaration is not the safety
net; the translator's honesty is, and every check in this library sits downstream of it.**

### 12.2 The repair

Three honest answers, in §6.1's order of preference: read it properly, declare `Card.ANY`,
or say you could not read it. This benchmark needs the third — the field this translator
knows is not in the file, and inventing a reading of `loose_eval` is how a reader ends up
confidently wrong about a producer it has never seen.

```python
def ns_outputs(src):
    for doc in src.docs(0):
        for line, rec, err in doc.jsonl():
            key = (f"{src.site.row}/{doc.name}/{line}",)
            if err:
                yield Unreadable(question="rollout.score",
                                 where=f"{doc.name} line {line}", detail=err)
                continue
            yield Answer("rollout.stop_reason", rec.get("finish_reason"), key=key)
            if "symbolic_correct" not in rec:
                yield Unreadable(
                    question="rollout.score", where=f"{doc.name} line {line}",
                    detail="no symbolic_correct field; this benchmark scores with "
                           "loose_eval and this translator does not read it")
                continue
            yield Answer("rollout.score", 1.0 if rec["symbolic_correct"] else 0.0, key=key)

NS_OUTPUTS_V2 = Input(
    name="outputs", via=ns_outputs, reads=(At(NS_TASK, "output*.jsonl"),),
    answers={"rollout.score": Card.ONE(because="every line carries a verdict"),
             "rollout.stop_reason": Card.ONE(because="finish_reason is on every record")})
FIXED = Reader(
    name="nemo-skills-lesson-v2", counts=Population.COMPLETED,
    landmarks=(NS_RUN, NS_TASK), inputs=(NS_CONFIG, NS_TOTALS, NS_OUTPUTS_V2),
    questions=NS_QUESTIONS, schema=NS_SCHEMA,
    rows=(Rows(question=Grain.ROLLOUT, may_create=("outputs",),
               because="the output*.jsonl lines ARE the enumeration of scored samples"),),
    attach=NS_READER.attachments, ignored=NS_READER.ignored)

good = FIXED.read(IFBENCH)
print("ok:", good.ok, " rollouts:", len(good.keys_of("rollout.score")))
print("unreadable:", len(good.of_kind(Kind.UNREADABLE)),
      " missing:", len(good.of_kind(Kind.MISSING)))
print(good.of_kind(Kind.MISSING)[0].detail[:96])
```

```text
ok: False  rollouts: 0
unreadable: 31  missing: 31
outputs.rollout.score at ('ifbench/output-rs0.jsonl/1',): expected exactly one, found 0
```

Thirty-one losses, named one by one, and the benchmark is no longer scored zero — it is
not scored at all, which is the truth.

Two mechanisms are firing there and they are worth telling apart:

- The **31 `UNREADABLE`** are yours. You said, per record, that it was there and you could
  not read it.
- The **31 `MISSING`** are the library's, and they are the §6.2 rule doing something it
  could not do until recently. `rollout.stop_reason` still names every row, so the row key
  exists; the key universe for a declared count is **the grain, not the question**, so
  `Card.ONE` on `rollout.score` is asked about every row this input named — including the
  ones it skipped. Before that change, a count was checked only against the answers that
  arrived, which made `Card.ONE` a claim about the answers rather than about the folder,
  and a translator could drop a record with no declaration anywhere noticing.

You would then either teach the translator `loose_eval` and delete both, or move
`rollout.score` to `Card.ANY` with a reason naming ifbench. Both are decisions; neither is
silence.

### 12.3 The number the reader still did not check

`metrics.json` reports `num_entries: 14042` for `mmlu` and `output.jsonl` holds 30 lines.
The corpus folder is a sample. Nothing above compares the two numbers, and every check in
this library reads exactly one question — so for a long time there was no way to.

`Compare` is the one declaration that reads two. It is the whole of the fix for a
measured read that reported **`ok: True, quiet: True, findings: 0`** while its own output
held `run.n_expected = 2500` one column from `run.n_scored = 3`, and a pass@1 of 0.6292
over the 2,497 rollouts it did not have:

```python
COMPARE_EXAMPLE = Compare(
    left="run.n_scored", right="run.n_expected", must="equal",
    because="this producer writes one results line per attempted rollout, so the "
            "manifest's count and the enumeration are two measurements of one number")
print(COMPARE_EXAMPLE.left, COMPARE_EXAMPLE.must, COMPARE_EXAMPLE.right)
```

It goes on the `Reader` as `compare=(...)`, because the reader is the only object that
holds both answers: they come from two inputs reading two files, and a translator is
handed its own input's files and has no method that reaches another's. `must` is one of
`equal`, `at_least`, `at_most`, written as a word rather than an operator because `<=` in
a declaration reads as an expression and this is a claim somebody is making about a
producer. A relation that does not hold is a `MISSING`, which fails `ok`; a side that
answered nothing is `UNCHECKED`, which does not, because the comparison did not run.

To use it here you would need both numbers as questions — the published count, and a
`Fold(op=Op.COUNT)` over the rollouts you read. Two questions, not one, because an
aggregate over a population may not share a name with a number stated in a document; §7
explains why the library refuses that.

---

## 13. How a read actually runs

```text
1. resolve landmarks              parents before children; a reader claims the folder
                                  when every landmark resolves in the number declared
2. run every input                any order — no input sees another's output
3. name every row                 (input, input_key), counted inside one input
4. settle each question           one claimant: done
                                  Merge.ONE:   the Order picks, the loser is recorded
                                  Merge.UNION: all land, deduped on identity
5. attach                         joins resolve here, after everyone has run
6. assign positions               sort by order_by, ties broken by (input, input_key)
7. fold up                        one rung at a time, lowest grain first
8. account                        cardinalities, mints, declared comparisons, and every
                                  file in the tree
```

Step 8 has one thing worth naming: a count is checked **once per site that exists**, so a
landmark that resolves to *zero* sites used to take every count underneath it with it, in
silence. Measured over the 53 corpus folders: **286 written `Card.ONE`/`Card.AT_LEAST_ONE`
claims never evaluated across 37 folders, against 285 `MISSING` findings actually
emitted** — the library skipped as many checks as it ran and said nothing about either
fact. It now emits one `UNCHECKED` naming the landmark and every count it took with it.
That is not a loss (your landmark's own `Card.ANY` said in writing this may happen, and an
absent directory holds no rows to have lost) but it costs `quiet`, and it is the list of
things the read is **not** claiming about that folder.

Two things fall out, and they are the point: **reading one folder twice gives a
byte-identical answer**, including the order of the complaints, and **adding an input does
not renumber the rows that were already there.**

---

## 14. What a read gives you back

```python
print("claimed:", result.claimed, " ok:", result.ok, " quiet:", result.quiet)
print("one value:", result.value("rollout.score", ("mmlu/0",)))
print("one stream:", len(result.stream("run.metrics")))
print("findings by kind:", {k.name: len(result.of_kind(k)) for k in Kind})
print("sites:", {n: len(s) for n, s in result.sites.items()})
print("hidden:", sorted(result.hidden))
```

`result.to_dict()` is a deterministic rendering of everything the read decided — two reads
of one folder produce equal dicts, so it is what you diff when you change a reader.

### Debugging a reader locally

To see the same thing laid out, with the traces it produced:

```sh
bazel run //local:trace -- /path/to/run-folder     # then open http://127.0.0.1:8765/#/debug
```

The page lists the reader that claimed the folder, seconds per input, every finding, and the
loss account (what had no column to land in, per question); each rollout links into the trace
view the service itself uses, whose provenance rail names the input, file and line behind each
fact. Nothing leaves your machine: the folder is decoded in this process into a temporary store
(`local/README.md`, "Debugging a reader").

---

## 15. Names this tutorial did not need

A name nobody is taught is a name nobody uses, so here is the rest of the export list,
with what each is for.

Of the 30 you type, this page typed every one except three:

| name | what it is for |
|---|---|
| `Translator` | the type alias `Callable[[Sources], Iterable[Answer]]`. For your annotations; never called. |
| `Sources` | the type of the object your translator is handed (§4.5). You use it constantly and rarely name it. |
| `Lane` | one ordered sequence inside a stream row — a trace, a sub-agent's turns, a monitor — declared on `Reader(lanes=)` and named by `Answer(lane=)`. Rows are deduplicated and numbered per lane; an answer with no lane is in the one unnamed lane. `../readers/harbor_trials/declare.py` declares several. |

Of the other 21, these come back from a read and you will handle them:

| name | what it is |
|---|---|
| `ReadResult` | what `.read()` returns (§14) |
| `Row` | one settled row: value, name, notes, position |
| `Edge` | one relation, or the record of why one end has no name |
| `Given` | an `Answer` after the runtime stamped provenance on it |
| `Finding` | one complaint: kind, question, where, detail, source |
| `Kind` | the closed list of eight finding kinds |
| `LOSS` | the five kinds that fail `ok`. `CONTESTED` and `ORPHAN` are outside it because an `Order` firing and a join recording its miss are those features working; `UNCHECKED` is outside it because it reports that something was never looked at, not that something went wrong. It still costs `quiet`. |
| `Site` | one resolved landmark instance — `.path`, `.row`, `.last_segment`, `.anchor`, `.matched`, `.chain` |
| `Refusal` | the one exception raised on purpose |

These are the trees a reader can be pointed at:

| name | what it is |
|---|---|
| `Tree` | the protocol: `paths()` and `read()`, and deliberately nothing else |
| `LocalTree` | a directory on disk. `.read("path/to/dir")` builds one for you |
| `Index` | the directory and file index built over one tree |
| `Entry` | one file in it: path and size |

And these are fixed lists you can read but not change:

| name | what it is |
|---|---|
| `TABLES` | the ten tables the bundled corpus readers land in. A **default, not a law** — pass `schema=` and this is not consulted |
| `DEPTH` | grain → row-key length (§4.3) |
| `PARENT` | grain → the grain above it, which is what a `Fold` must lift into |
| `LOWEST_FIRST` | the order the merge walks the ladder in |
| `AGGREGATE_OPS` | the five `Op`s whose value is a property of the population (§7) |
| `ROOT` | the landmark that is always there: the tree root (§5.1) |
| `resolve_landmarks` | landmark resolution on its own, for debugging a `where` (§4.6) |
| `__version__` | the library version, as a string |

```python
print(len(TABLES), "tables:", TABLES[:4], "...")
print("PARENT:", {k.name: v.name for k, v in PARENT.items()})
print("LOWEST_FIRST:", [g.name for g in LOWEST_FIRST])
```

---

## 16. Mistakes people make the first time

- **Keying on `site.last_segment` where `where` is more than one segment deep.** Use `site.row`.
  This is the single most common way to silently merge two rows into one, and it has cost
  four independent authors real time.
- **Yielding `None` and calling it an answer.** `d.get("score")` turns "we do not know"
  into a null the database keeps and nobody queries. It no longer satisfies a demanded
  count. Decide: read it properly, declare `Card.ANY`, or yield `Unreadable`.
- **Running a value through a helper that answers a slightly different question.** A float
  helper that rejects booleans on purpose destroyed **100% of one benchmark's scores**, and
  every read reported `ok`. `splitlines()` is not `split("\n")`. Check what your helper
  refuses.
- **Skipping the error branch of `.jsonl()`.** Yield `Unreadable`. The library will find
  the shortfall anyway, but its message can only name the line; yours can name the record.
- **Never yielding `Unreadable` at all.** If a record is there and you cannot read it, that
  is what the word is for.
- **Reaching for `src.doc(0)` where the file is sometimes absent.** It reports the miss and
  gives up on that site, which is contained but is still a site that produced nothing. If
  the file is honestly optional, `first(0)` or `has(0)` says so.
- **Believing `Card.ONE` is a claim about the folder.** It is a claim about the rows your
  input named at that grain. It cannot see a record you dropped whole, naming nothing about
  it under any question — for that you need a denominator the producer published and a
  `Compare`.
- **Forgetting that a `Merge.UNION` row is a mapping and nothing checks its cells.** A row
  of four columns, three of them null, passes every check in this library. See the README's
  "What is here and what is not"; it is measured, named and unbuilt.
- **Reading `ok` as "correct".** It means every file was named, every declared count was
  met, every declared join and fold came up full, and every declared `Compare` held. A
  translator reading the wrong field produces a green read, and that is not a hypothetical
  — it is §12.1.
- **Treating a missing file as zero.** `d.get("score", 0.0)` turns "we do not know" into
  "it scored nothing".
- **`Card.ONE` on a file that is sometimes absent.** The count is checked at **every** site
  of the landmark, including the ones where your file does not exist.
- **Putting a decision in the translator.** `if a is None: a = b` is an `Order`, written
  where no reviewer will find it.
- **Searching instead of addressing.** Your translator gets the files its declaration
  matched and has no method that reaches another. Need a second file? Declare a second
  `At`, or a second `Input`.
- **Handing a key space several ids at once.** `keys={"event": (a, b, c)}` is refused. One
  answer names one thing; three references are three answers with the same row key.
- **Designing to avoid findings.** `CONTESTED` and `ORPHAN` do not fail `ok`. If you are
  dropping a `Relate` to keep a read clean, you are optimising the wrong number.
- **A `because=` that restates the code.** `because="reads the manifest"` is worthless.
  Say why it is true, ideally with a number and a file path.
- **Reusing a declaration from another producer without checking what it assumed.** If it
  was written where one directory is one rollout and your folder has an extra level, you
  get two rows per rollout and no complaint.

---

## 17. Checklist for a folder you have never seen

1. **List every file.** `find <folder> -type f`. Not the first hundred — every one, because
   every one needs an input or an `Ignore`.
2. **Find the marker.** Which file is always present and says "this is my producer"? Does
   it sit in one place, or two across versions?
3. **Find the enumeration.** What is one row per case — a directory each, or a line each in
   one file? Decide now whether crashed cases count, because `counts=` will ask.
4. **Decide the row key.** If your `where` is more than one segment deep, it is `site.row`.
   If the records carry no id, it is the line number inside something named.
5. **Write the questions before the translators.** Then hand the `Reader` your real
   `schema=` and let it refuse the landings that do not exist.
6. **Run it and read the refusals.** Each one prints the line to write. Expect three or
   four before the reader constructs at all.
7. **Find the producer's own count** — `n_results`, `num_entries`, `trials`, whatever it
   is called. Store it as its own question and write the `Compare` against your
   enumeration. It is the only check in here that can notice a record you never saw.
8. **Read `out.report()`.** Aim for `ok`. `quiet` is a bonus, not a target — but read the
   `UNCHECKED` lines before you move on: they are the declarations this folder gave the
   library no chance to run.
9. **Run it on a second folder from the same producer.** This is where you find out what
   you assumed. Every one of the readers written against this page that turned out to be
   confidently wrong was wrong on the folder next door, not on the one it was written for.
10. **Count something outside the library and compare.** `ok` claims your declarations
   held. It does not claim your translator read the right field, and it cannot: a
   translator is ordinary Python, which is the whole point of it being ordinary Python.

---

## 18. Where to look next

- `tests/examples/tiny_reader.py` — one runnable reader with every feature
  in it, including the ones this page only described.
- `../readers/` — one reader per real producer (`dispatch.py` orders them), and
  `store_schema.py` for how a real `schema=` and its `NO_HOME` list are kept.
- `README.md` — the reference: every construct, every rule that can refuse you. Changing a
  reader that exists: *Changing a reader* there, before you write anything.
- The refusal messages themselves — each names the line to write. Often the fastest answer.
