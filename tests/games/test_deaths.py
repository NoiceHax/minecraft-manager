"""The vanilla death table: coverage, ordering, and the adversarial-name cases."""

from __future__ import annotations

import re

import pytest

from mcmanager.games.minecraft.deaths import (
    DEATH_PATTERNS,
    DEATH_TEMPLATES,
    compile_template,
    match_death,
)

_PLACEHOLDER_RE = re.compile(r"%[123]\$s")


def _render(
    template: str,
    player: str,
    killer: str = "Zombie",
    item: str = "Sharpness Sword",
) -> str:
    """Turn a template back into a concrete message, the way the game would."""
    return template.replace("%1$s", player).replace("%2$s", killer).replace("%3$s", item)


# --------------------------------------------------------------------------- table integrity


def test_the_table_is_substantial() -> None:
    """A thin table silently degrades into "everything is a ConsoleLog"."""
    assert len(DEATH_PATTERNS) >= 100


def test_every_template_has_a_player_placeholder() -> None:
    for template in DEATH_TEMPLATES:
        assert "%1$s" in template, template


def test_no_template_uses_item_without_killer() -> None:
    """``%3$s`` never appears alone: every vanilla ``.item`` variant also names the killer."""
    for template in DEATH_TEMPLATES:
        if "%3$s" in template:
            assert "%2$s" in template, template


def test_patterns_are_sorted_by_literal_length_descending() -> None:
    """The property that makes hand-ordering unnecessary."""
    lengths = [len(_PLACEHOLDER_RE.sub("", template)) for template, _ in DEATH_PATTERNS]
    assert lengths == sorted(lengths, reverse=True)


def test_duplicate_templates_are_compiled_once() -> None:
    """``death.attack.mob`` and ``death.attack.player`` are the same English string."""
    templates = [template for template, _ in DEATH_PATTERNS]
    assert len(templates) == len(set(templates))
    assert "%1$s was slain by %2$s" in templates


def test_every_template_round_trips_through_its_own_pattern() -> None:
    """Generate the message the game would print, and require the table to match it back.

    This is what makes a typo in a checked-in template visible: a template with a stray double
    space renders a message that its own regex still matches, but the *table* then matches a
    different, shorter pattern first, and the assertion on ``template`` fails.
    """
    for template in DEATH_TEMPLATES:
        message = _render(template, "Steve")
        found = match_death(message)
        assert found is not None, f"no match for {message!r} from {template!r}"
        assert found.player == "Steve", (message, found)
        assert found.template == template, (message, found.template, template)


# ---------------------------------------------------------------------- real sampled deaths


@pytest.mark.parametrize(
    ("message", "player", "killer", "template"),
    [
        # Every one of these is a verbatim line from tests/fixtures/logs/archives/.
        (
            "Hypixelite was blown up by Creeper",
            "Hypixelite",
            "Creeper",
            "%1$s was blown up by %2$s",
        ),
        (
            "Hypixelite was poked to death by a sweet berry bush",
            "Hypixelite",
            None,
            "%1$s was poked to death by a sweet berry bush",
        ),
        ("Hypixelite drowned", "Hypixelite", None, "%1$s drowned"),
        ("bharath_720 was shot by Skeleton", "bharath_720", "Skeleton", "%1$s was shot by %2$s"),
        ("Hypixelite was slain by Zombie", "Hypixelite", "Zombie", "%1$s was slain by %2$s"),
        (
            "bharath_720 fell from a high place",
            "bharath_720",
            None,
            "%1$s fell from a high place",
        ),
        (
            "Hypixelite tried to swim in lava",
            "Hypixelite",
            None,
            "%1$s tried to swim in lava",
        ),
        (
            "Hypixelite tried to swim in lava to escape Skeleton",
            "Hypixelite",
            "Skeleton",
            "%1$s tried to swim in lava to escape %2$s",
        ),
    ],
)
def test_real_sampled_death_lines(
    message: str, player: str, killer: str | None, template: str
) -> None:
    found = match_death(message)
    assert found is not None
    assert found.player == player
    assert found.killer == killer
    assert found.template == template


# ------------------------------------------------------------------------------- ordering


def test_longer_literal_wins_over_the_prefix_it_contains() -> None:
    """``...whilst trying to escape X`` beats the bare form, with no hand-ordering."""
    escaped = match_death("Steve drowned whilst trying to escape Drowned")
    assert escaped is not None
    assert escaped.template == "%1$s drowned whilst trying to escape %2$s"
    assert escaped.killer == "Drowned"

    plain = match_death("Steve drowned")
    assert plain is not None
    assert plain.template == "%1$s drowned"
    assert plain.killer is None


def test_item_variant_wins_over_the_killer_only_variant() -> None:
    found = match_death("Steve was slain by Zombie using Sharpness V Netherite Sword")
    assert found is not None
    assert found.template == "%1$s was slain by %2$s using %3$s"
    assert found.killer == "Zombie"
    assert found.item == "Sharpness V Netherite Sword"


def test_killer_capture_splits_at_the_first_using() -> None:
    """An ambiguous message has to resolve one way; lazy groups pick the likelier reading.

    ``Zombie`` wielding an item called ``a bow using Bow`` beats a mob literally named
    ``Zombie using a bow``. Asserted so the choice is a decision rather than an accident.
    """
    found = match_death("Steve was slain by Zombie using a bow using Bow")
    assert found is not None
    assert found.killer == "Zombie"
    assert found.item == "a bow using Bow"


def test_thorns_item_template_binds_groups_positionally() -> None:
    """``%1$s was killed by %3$s while trying to hurt %2$s`` names the item before the killer."""
    found = match_death("Steve was killed by Thorns III Chestplate while trying to hurt Alex")
    assert found is not None
    assert found.item == "Thorns III Chestplate"
    assert found.killer == "Alex"


def test_intentional_game_design_beats_the_generic_killed_by() -> None:
    found = match_death("Steve was killed by [Intentional Game Design]")
    assert found is not None
    assert found.template == "%1$s was killed by [Intentional Game Design]"
    assert found.killer is None


def test_magic_variants_do_not_shadow_each_other() -> None:
    plain = match_death("Steve was killed by magic")
    indirect = match_death("Steve was killed by Witch using magic")
    with_item = match_death("Steve was killed by Witch using Splash Potion")
    assert plain is not None
    assert plain.template == "%1$s was killed by magic"
    assert indirect is not None
    assert indirect.template == "%1$s was killed by %2$s using magic"
    assert indirect.killer == "Witch"
    assert with_item is not None
    assert with_item.template == "%1$s was killed by %2$s using %3$s"


# ------------------------------------------------------------------------- adversarial names


def test_a_player_whose_name_embeds_a_death_fragment() -> None:
    """The ``Notch fell`` case from the plan, in the only form the game can actually produce.

    A Minecraft account name is ``[A-Za-z0-9_]{3,16}``: it cannot contain a space, so a player
    literally called ``Notch fell`` does not exist and the anchored name class makes that
    structural rather than hopeful. The reachable version of the same hazard is a name that
    embeds a death-message word with an underscore.
    """
    found = match_death("Notch_fell fell from a high place")
    assert found is not None
    assert found.player == "Notch_fell"
    assert found.template == "%1$s fell from a high place"


def test_a_name_with_a_space_cannot_be_captured() -> None:
    """The structural guarantee, asserted rather than assumed."""
    assert match_death("Notch fell fell from a high place") is None


def test_a_killer_named_after_a_player_does_not_confuse_the_player_group() -> None:
    found = match_death("Steve was slain by Notch_fell")
    assert found is not None
    assert found.player == "Steve"
    assert found.killer == "Notch_fell"


@pytest.mark.parametrize(
    "message",
    [
        "[13:24:37] [Server thread/INFO]: Steve drowned",
        " Steve drowned",
        "Steve drowned ",
        "Steve  drowned",
        "SteveWithAVeryLongNameIndeed drowned",  # 27 characters, over the 16 limit
        "",
        "drowned",
    ],
)
def test_unanchored_and_malformed_input_does_not_match(message: str) -> None:
    """No prefix, no padding and no over-long name may sneak through.

    The first case is the old script's bug expressed as a death: an unanchored pattern would
    capture ``[13:24:37] [Server thread/INFO]: Steve`` as the player.
    """
    assert match_death(message) is None


def test_match_death_never_raises_on_hostile_input() -> None:
    for hostile in ("\x00", "\ud800", "a" * 100_000, "Steve drowned\n\nmore"):
        match_death(hostile)


def test_compile_template_escapes_regex_metacharacters() -> None:
    """``[Intentional Game Design]`` contains a character class if you forget to escape it."""
    pattern = compile_template("%1$s was killed by [Intentional Game Design]")
    assert pattern.match("Steve was killed by [Intentional Game Design]") is not None
    assert pattern.match("Steve was killed by I") is None
