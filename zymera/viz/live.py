"""
Interactive matplotlib keyboard teleop window.

Open a window, press a key (or click a widget), the selected agent
steps, the canvas redraws. Multi-agent with on-screen toggles for grid /
comm / fog. Same renderer as :func:`zymera.render_gif`.

Keys (movement applies to the *selected* agent; others STAY):

    w / k / ↑     →  NORTH
    s / j / ↓     →  SOUTH
    a / h / ←     →  WEST
    d / l / →     →  EAST
    space         →  all STAY

    tab           cycle selected agent
    0–9           jump to agent N

    q / escape    close window, return trajectory

Widgets at the bottom: Grid + Comm checkboxes, Fog mode radio
(off/self/all), Controlled-agent radio.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp

from ..env import ActionId
from ..helper import Helper


# ---- Key → action mapping --------------------------------------------------

KEY_TO_ACTION: Dict[str, ActionId] = {
    "w": ActionId.NORTH, "k": ActionId.NORTH, "up":    ActionId.NORTH,
    "s": ActionId.SOUTH, "j": ActionId.SOUTH, "down":  ActionId.SOUTH,
    "a": ActionId.WEST,  "h": ActionId.WEST,  "left":  ActionId.WEST,
    "d": ActionId.EAST,  "l": ActionId.EAST,  "right": ActionId.EAST,
    " ": ActionId.STAY,
}

_QUIT_KEYS = ("q", "escape")
_FOG_CYCLE = ("off", "self", "all")

# matplotlib default keymaps that conflict with our movement / toggle keys.
# Cleared at the top of live_keyboard() so the env keys aren't shadowed by
# fullscreen / save / grid / scale / pan / etc.
_KEYMAP_BLOCK = (
    "keymap.fullscreen", "keymap.home", "keymap.back",
    "keymap.forward", "keymap.pan", "keymap.zoom",
    "keymap.save", "keymap.help", "keymap.quit",
    "keymap.grid", "keymap.grid_minor",
    "keymap.yscale", "keymap.xscale", "keymap.copy",
)


def _disable_matplotlib_keymaps() -> None:
    """Clear rcParams keymap entries that conflict with the env keys."""
    import matplotlib.pyplot as plt
    for key in _KEYMAP_BLOCK:
        plt.rcParams[key] = []


# ---- mutable state shared with callbacks ----------------------------------


@dataclass
class _LiveState:
    """State shared with matplotlib key + widget callbacks."""

    world: object
    n_agents: int
    fog_radius: int
    step: int = 0
    last: str = "—"
    selected: int = 0
    show_grid: bool = True
    show_comm: bool = True
    fog_mode: str = "off"


# ---- main entry point ------------------------------------------------------


def live_keyboard(
    grid_h: int = 10,
    grid_w: int = 10,
    n_agents: int = 1,
    comm_radius: int = 2,
    fog_radius: int = 3,
    seed: int = 0,
) -> dict:
    """Open an interactive matplotlib window driving up to N agents.

    Returns the recorded trajectory (dict of lists) when the user
    quits or closes the window. Shape matches the per-step structure
    of :func:`zymera.rollout` — viz helpers consume either.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons, RadioButtons

    from ..env import World
    from .render import (
        compute_fog_mask, draw_comm_graph, draw_fog, draw_frame,
        draw_grid_lines, draw_selection_highlight,
    )

    _disable_matplotlib_keymaps()

    world = World.initial(
        jax.random.PRNGKey(seed), grid_h, grid_w, n_agents,
    )
    helper = Helper(comm_radius=comm_radius)

    state = _LiveState(world=world, n_agents=n_agents, fog_radius=fog_radius)
    trajectory: dict = {
        "world":  [world],
        "obs":    [world.sense()],
        "action": [],
        "reward": [],
        "done":   [],
    }

    # ---- figure layout ---------------------------------------------------

    fig = plt.figure(figsize=(9, 9))
    try:
        fig.canvas.manager.set_window_title("zymera-keys")
    except Exception:
        pass

    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[8.0, 1.5],
        width_ratios=[1.0, 1.0, 1.5],
        hspace=0.18, wspace=0.08,
    )
    ax       = fig.add_subplot(gs[0, :])
    ax_check = fig.add_subplot(gs[1, 0])
    ax_fog   = fig.add_subplot(gs[1, 1])
    ax_agent = fig.add_subplot(gs[1, 2])

    for sub_ax, label in (
        (ax_check, "Overlays"),
        (ax_fog,   "Fog of war"),
        (ax_agent, "Controlled agent"),
    ):
        sub_ax.set_xticks([]); sub_ax.set_yticks([])
        sub_ax.set_title(label, fontsize=9)
        for spine in sub_ax.spines.values():
            spine.set_color("0.7")

    check = CheckButtons(
        ax_check, ["Grid", "Comm"],
        actives=[state.show_grid, state.show_comm],
    )
    radio_fog = RadioButtons(
        ax_fog, _FOG_CYCLE, active=_FOG_CYCLE.index(state.fog_mode),
    )
    agent_labels = tuple(f"agent {i}" for i in range(n_agents))
    radio_agent = RadioButtons(ax_agent, agent_labels, active=state.selected)

    widgets = (check, radio_fog, radio_agent)   # keep refs alive

    # ---- redraw composer --------------------------------------------------

    def _title() -> str:
        cov = float(jnp.mean(state.world.visited))
        return (
            f"step {state.step}   last: {state.last}   coverage: {cov:.2f}   "
            f"  (tab / 0–9 to switch agent · q to quit)"
        )

    def _redraw() -> None:
        H, W = state.world.visited.shape
        draw_frame(ax, state.world, title=None)
        if state.fog_mode != "off":
            mask = compute_fog_mask(
                state.world, state.fog_mode, state.fog_radius, state.selected,
            )
            draw_fog(ax, mask)
        if state.show_comm and state.n_agents > 1:
            draw_comm_graph(ax, state.world, helper.comm(state.world))
        if state.show_grid:
            draw_grid_lines(ax, H, W)
        if state.n_agents > 1:
            draw_selection_highlight(ax, state.world, state.selected)
        ax.set_title(_title(), fontsize=10, loc="left")
        fig.canvas.draw_idle()

    # ---- widget callbacks -------------------------------------------------

    def _on_check(label: str) -> None:
        if label == "Grid":
            state.show_grid = not state.show_grid
        elif label == "Comm":
            state.show_comm = not state.show_comm
        _redraw()

    def _on_fog(label: str) -> None:
        state.fog_mode = label
        _redraw()

    def _on_agent(label: str) -> None:
        state.selected = int(label.split()[1])
        _redraw()

    check.on_clicked(_on_check)
    radio_fog.on_clicked(_on_fog)
    radio_agent.on_clicked(_on_agent)

    # ---- keyboard ---------------------------------------------------------

    def _set_selected(i: int) -> None:
        state.selected = i % state.n_agents
        radio_agent.set_active(state.selected)

    def _on_key(event) -> None:
        key = event.key
        if key in _QUIT_KEYS:
            plt.close(fig); return

        if key == "tab":
            _set_selected(state.selected + 1); _redraw(); return
        if key is not None and key.isdigit():
            i = int(key)
            if 0 <= i < state.n_agents:
                _set_selected(i); _redraw()
            return

        action_id: Optional[ActionId] = KEY_TO_ACTION.get(key)
        if action_id is None:
            return

        actions = [int(ActionId.STAY)] * state.n_agents
        actions[state.selected] = int(action_id)
        action = jnp.array(actions, dtype=jnp.int32)

        out = state.world.step(action)
        state.world = out.world
        state.step += 1
        state.last = f"agent {state.selected}: {action_id.name}"

        trajectory["action"].append(action)
        trajectory["world"].append(out.world)
        trajectory["obs"].append(out.obs)
        trajectory["reward"].append(out.reward)
        trajectory["done"].append(out.done)
        _redraw()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _redraw()
    plt.show(block=True)

    del widgets
    return trajectory


# ---- CLI entry point (zymera-keys) ----------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zymera-keys",
        description=(
            "Multi-agent keyboard teleop in a matplotlib window. "
            "WASD / vi-keys / arrows drive the selected agent; tab / 0-9 "
            "to switch; widget toggles for grid / comm / fog; q or esc to quit."
        ),
    )
    parser.add_argument("--grid-h",      type=int, default=10)
    parser.add_argument("--grid-w",      type=int, default=10)
    parser.add_argument("--n-agents",    type=int, default=1)
    parser.add_argument("--comm-radius", type=int, default=2)
    parser.add_argument("--fog-radius",  type=int, default=3)
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()
    live_keyboard(
        grid_h=args.grid_h, grid_w=args.grid_w, n_agents=args.n_agents,
        comm_radius=args.comm_radius, fog_radius=args.fog_radius,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
