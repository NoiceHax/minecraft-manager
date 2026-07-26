"""The vanilla death-message table.

Every concrete vanilla death string, checked in **verbatim** as its ``%1$s``/``%2$s``/``%3$s``
template exactly as it appears in ``en_us.json``, and compiled to an anchored regex at import.

Three properties make this table safe rather than a pile of guesses:

1. **Verbatim templates, not hand-written regexes.** The strings are copied, the regex is
   generated. A typo in a hand-written pattern is invisible; a typo in a template is a string that
   never matches a real line and shows up in the unrecognised-line ratio.
2. **Anchored at both ends.** ``^`` and ``$`` around every compiled pattern, with ``%1$s`` bound
   to :data:`~mcmanager.games.minecraft.patterns.PLAYER_NAME` rather than ``.+``. A death line for
   a player called ``Notch_fell`` therefore yields ``Notch_fell``, and no amount of greed can walk
   a capture group backwards into the log prefix.
3. **Sorted by literal length descending.** ``%1$s walked into a cactus whilst trying to escape
   %2$s`` is tried before ``%1$s was pricked to death``, and ``%1$s was slain by %2$s using %3$s``
   before ``%1$s was slain by %2$s``, with nobody hand-ordering the table. Anchoring alone already
   settles most of these; the sort settles the rest, where a lazy ``%2$s`` could otherwise swallow
   the suffix of a longer template.

A first-match linear scan over ~100 patterns is only ever reached by a line that matched nothing
else, so its cost is irrelevant. It is measured anyway by
:func:`mcmanager.games.minecraft.parser.scan`.

**Tier-2 fallback for plugin and modded deaths lives in the log pipeline, not here**: a
``ConsoleLog`` whose message begins with a name currently in the online set is upgraded to
``PlayerDeath(template=None)`` and logged under ``mcmanager.deaths.unmatched`` so this table can
grow. That is the clean answer to "the parser cannot know who is online" - it does not need to.
The enricher does.

The half of that fallback which *is* game knowledge lives here, as :func:`looks_like_death`: the
remainder of the line has to begin the way the game itself begins a death message. Without it the
online-name test alone is not a test at all, because ``online-mode=false`` lets anybody log in as
``Saving`` - and the real archives in this repo contain ``Saving players``, ``Saving chunks for
level 'ServerLevel[world]'/minecraft:overworld``, ``Preparing spawn area: 2%``, ``Time elapsed:
123 ms`` and ``Closing Session ...``, every one of which begins with a legal player name. A
player picking one of those names would otherwise fabricate three deaths into Discord on every
autosave, and corrupt the session summary's death count with them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mcmanager.games.minecraft.patterns import PLAYER_NAME

__all__ = [
    "DEATH_LEAD_WORDS",
    "DEATH_PATTERNS",
    "DEATH_TEMPLATES",
    "DeathMatch",
    "compile_template",
    "looks_like_death",
    "match_death",
]

DEATH_TEMPLATES: tuple[str, ...] = (
    # -- death.attack.* : killed by something ------------------------------------------------
    "%1$s was squashed by a falling anvil",
    "%1$s was squashed by a falling anvil whilst fighting %2$s",
    "%1$s was shot by %2$s",
    "%1$s was shot by %2$s using %3$s",
    "%1$s was killed by %2$s",  # death.attack.badRespawnPoint, %2$s = [Intentional Game Design]
    "%1$s was pricked to death",
    "%1$s walked into a cactus whilst trying to escape %2$s",
    "%1$s was squished too much",
    "%1$s was squashed by %2$s",
    "%1$s was roasted in dragon's breath",
    "%1$s was roasted in dragon's breath by %2$s",
    "%1$s drowned",
    "%1$s drowned whilst trying to escape %2$s",
    "%1$s died from dehydration",
    "%1$s died from dehydration whilst trying to escape %2$s",
    "%1$s was killed by even more magic",
    "%1$s blew up",
    "%1$s was blown up by %2$s",
    "%1$s was blown up by %2$s using %3$s",
    "%1$s was killed by [Intentional Game Design]",
    "%1$s hit the ground too hard",
    "%1$s hit the ground too hard whilst trying to escape %2$s",
    "%1$s was squashed by a falling block",
    "%1$s was squashed by a falling block whilst fighting %2$s",
    "%1$s was skewered by a falling stalactite",
    "%1$s was skewered by a falling stalactite whilst fighting %2$s",
    "%1$s was fireballed by %2$s",
    "%1$s was fireballed by %2$s using %3$s",
    "%1$s went off with a bang",
    "%1$s went off with a bang due to a firework fired from %3$s by %2$s",
    "%1$s went off with a bang whilst fighting %2$s",
    "%1$s experienced kinetic energy",
    "%1$s experienced kinetic energy whilst trying to escape %2$s",
    "%1$s froze to death",
    "%1$s was frozen to death by %2$s",
    "%1$s died",
    "%1$s died because of %2$s",
    "%1$s was killed",
    "%1$s was killed whilst fighting %2$s",
    "%1$s discovered the floor was lava",
    "%1$s walked into the danger zone due to %2$s",
    "%1$s went up in flames",
    "%1$s walked into fire whilst fighting %2$s",
    "%1$s suffocated in a wall",
    "%1$s suffocated in a wall whilst fighting %2$s",
    "%1$s was killed by %2$s using magic",
    "%1$s was killed by %2$s using %3$s",
    "%1$s tried to swim in lava",
    "%1$s tried to swim in lava to escape %2$s",
    "%1$s was struck by lightning",
    "%1$s was struck by lightning whilst fighting %2$s",
    "%1$s was smashed by %2$s",
    "%1$s was smashed by %2$s with %3$s",
    "%1$s was killed by magic",
    "%1$s was killed by magic whilst trying to escape %2$s",
    "%1$s was slain by %2$s",  # death.attack.mob and death.attack.player share this string
    "%1$s was slain by %2$s using %3$s",
    "%1$s burned to death",
    "%1$s was burned to a crisp whilst fighting %2$s",
    "%1$s was burned to a crisp whilst fighting %2$s wielding %3$s",
    "%1$s fell out of the world",
    "%1$s didn't want to live in the same world as %2$s",
    "%1$s left the confines of this world",
    "%1$s left the confines of this world whilst fighting %2$s",
    "%1$s was obliterated by a sonically-charged shriek",
    "%1$s was obliterated by a sonically-charged shriek whilst trying to escape %2$s",
    "%1$s was obliterated by a sonically-charged shriek whilst trying to escape %2$s wielding %3$s",
    "%1$s was impaled on a stalagmite",
    "%1$s was impaled on a stalagmite whilst fighting %2$s",
    "%1$s starved to death",
    "%1$s starved to death whilst fighting %2$s",
    "%1$s was stung to death",
    "%1$s was stung to death by %2$s",
    "%1$s was stung to death by %2$s using %3$s",
    "%1$s was poked to death by a sweet berry bush",
    "%1$s was poked to death by a sweet berry bush whilst trying to escape %2$s",
    "%1$s was killed while trying to hurt %2$s",
    "%1$s was killed by %3$s while trying to hurt %2$s",
    "%1$s was pummeled by %2$s",
    "%1$s was pummeled by %2$s using %3$s",
    "%1$s was impaled by %2$s",
    "%1$s was impaled by %2$s with %3$s",
    "%1$s was blown away by %2$s",
    "%1$s was blown away by %2$s using %3$s",
    "%1$s withered away",
    "%1$s withered away whilst fighting %2$s",
    "%1$s was shot by a skull from %2$s",
    "%1$s was shot by a skull from %2$s using %3$s",
    # -- death.fell.* : gravity ---------------------------------------------------------------
    "%1$s fell from a high place",
    "%1$s fell off a ladder",
    "%1$s fell off some vines",
    "%1$s fell off some weeping vines",
    "%1$s fell off some twisting vines",
    "%1$s fell off scaffolding",
    "%1$s fell while climbing",
    "%1$s fell out of the water",
    "%1$s was doomed to fall",
    "%1$s was doomed to fall by %2$s",
    "%1$s was doomed to fall by %2$s using %3$s",
    "%1$s fell too far and was finished by %2$s",
    "%1$s fell too far and was finished by %2$s using %3$s",
    # -- other vanilla death causes -----------------------------------------------------------
    "%1$s was struck down by %2$s",
    "%1$s was struck down by %2$s using %3$s",
)
"""Every vanilla death string, verbatim.

Duplicates are intentional in the source game (``death.attack.mob`` and ``death.attack.player``
are the same English string) and are de-duplicated once, at compile time, so the linear scan is
not lengthened by them.
"""

_PLACEHOLDER_RE = re.compile(r"%(?P<index>[123])\$s")

_GROUP_FOR_INDEX = {"1": "player", "2": "killer", "3": "item"}


def _literal_length(template: str) -> int:
    """How much fixed text a template asserts. The sort key, and the whole ordering strategy."""
    return len(_PLACEHOLDER_RE.sub("", template))


def compile_template(template: str) -> re.Pattern[str]:
    """Turn one ``%N$s`` template into an anchored regex with named groups.

    ``%1$s`` becomes :data:`~mcmanager.games.minecraft.patterns.PLAYER_NAME` - a character class
    that cannot contain a bracket, a colon or a space, so it cannot reach back into a log prefix.
    ``%2$s`` and ``%3$s`` become lazy ``.+?`` groups, because an entity or item name is arbitrary
    text. Laziness means ``%2$s using %3$s`` splits at the **first** ``" using "``, which is the
    right call: a mob named ``Zombie`` wielding a ``Bow using Bow`` is far likelier than a mob
    literally named ``Zombie using a bow``, and only one of the two readings can win.

    Positional order in the template is respected, which matters for exactly one string:
    ``%1$s was killed by %3$s while trying to hurt %2$s`` names the item before the killer.
    """
    parts: list[str] = []
    cursor = 0
    for placeholder in _PLACEHOLDER_RE.finditer(template):
        parts.append(re.escape(template[cursor : placeholder.start()]))
        index = placeholder["index"]
        if index == "1":
            parts.append(f"(?P<player>{PLAYER_NAME})")
        else:
            parts.append(f"(?P<{_GROUP_FOR_INDEX[index]}>.+?)")
        cursor = placeholder.end()
    parts.append(re.escape(template[cursor:]))
    return re.compile("^" + "".join(parts) + "$", re.DOTALL)


def _build() -> tuple[tuple[str, re.Pattern[str]], ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for template in DEATH_TEMPLATES:
        if template not in seen:
            seen.add(template)
            unique.append(template)
    # Descending literal length; the template string itself breaks ties so the order is stable
    # across interpreter runs and a golden file cannot flap.
    unique.sort(key=lambda t: (-_literal_length(t), t))
    return tuple((template, compile_template(template)) for template in unique)


DEATH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = _build()
"""``(template, compiled)`` pairs, in the order :func:`match_death` tries them."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DeathMatch:
    """One death line, decomposed.

    Attributes:
        player: The name captured by ``%1$s``.
        killer: ``%2$s`` - the entity or player responsible, when the template has one.
        item: ``%3$s`` - the named weapon, when the template has one.
        template: The verbatim vanilla template that matched. Carried onto
            :class:`~mcmanager.core.events.PlayerDeath` so that ``None`` there means
            unambiguously "this came from the pipeline's online-player fallback, not from the
            table".
    """

    player: str
    killer: str | None = None
    item: str | None = None
    template: str


def _lead_words() -> frozenset[str]:
    """Every first word a vanilla death message uses after the player's name.

    Derived from :data:`DEATH_TEMPLATES` rather than typed out, so a template added to the table
    widens the tier-2 gate for free and the two can never disagree.
    """
    words: set[str] = set()
    for template in DEATH_TEMPLATES:
        remainder = template.removeprefix("%1$s ")
        head = remainder.split(maxsplit=1)
        if head:
            words.add(head[0])
    return frozenset(words)


DEATH_LEAD_WORDS: frozenset[str] = _lead_words()
"""``was``, ``died``, ``fell``, ``blew``, ``went``, ``walked``, ``drowned``, ... - eighteen of them.

The gate the pipeline's tier-2 fallback applies to the *remainder* of a line that begins with an
online player's name. See :func:`looks_like_death`.
"""


def looks_like_death(remainder: str) -> bool:
    """Does the text after a player's name begin the way a death message begins?

    The narrowing that makes the tier-2 fallback safe. It deliberately asks only about the **first
    word**, because that is the part plugin and modded death messages inherit from the game -
    ``%1$s was disintegrated by a Void Reaver`` is a new template, not a new grammar - while it is
    the part server chatter never shares. ``Saving players``, ``Saving chunks for level ...`` and
    ``Preparing spawn area: 2%`` all fail it, and those are real lines from this deployment's
    archives that a player named ``Saving`` or ``Preparing`` would otherwise turn into deaths.

    The trade is explicit: a plugin whose death message opens with a word the game never uses is
    missed. The pipeline logs those under ``mcmanager.deaths.unmatched`` at DEBUG anyway, so the
    evidence for widening :data:`DEATH_TEMPLATES` is collected either way - and a missed death is
    a missing announcement, while a false one is a lie about a player, published to Discord and
    counted in their session summary.

    Args:
        remainder: The message with the leading player name and its following space removed.

    Returns:
        True if the first word is one the vanilla death table uses.
    """
    head = remainder.split(maxsplit=1)
    return bool(head) and head[0] in DEATH_LEAD_WORDS


def match_death(message: str) -> DeathMatch | None:
    """First-match linear scan of the death table. ``None`` if nothing matched.

    Pure and total. Called only after every other pattern has declined the line.
    """
    for template, pattern in DEATH_PATTERNS:
        found = pattern.match(message)
        if found is None:
            continue
        groups = found.groupdict()
        return DeathMatch(
            player=found["player"],
            killer=groups.get("killer"),
            item=groups.get("item"),
            template=template,
        )
    return None
