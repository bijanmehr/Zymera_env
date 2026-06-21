"""viz — render primitives + live keyboard pure parts."""

import matplotlib.pyplot as plt
import numpy as np

import zymera
from zymera.viz import compute_fog_mask, draw_comm_graph, draw_frame, draw_grid_lines
from zymera.viz.live import (
    _FOG_CYCLE, _KEYMAP_BLOCK, _QUIT_KEYS,
    _disable_matplotlib_keymaps, KEY_TO_ACTION,
)


# ---- keymap + fog mask -----------------------------------------------------


def test_key_map_covers_every_direction():
    expected = {zymera.ActionId.NORTH, zymera.ActionId.SOUTH,
                zymera.ActionId.EAST, zymera.ActionId.WEST,
                zymera.ActionId.STAY}
    assert expected <= set(KEY_TO_ACTION.values())


def test_wasd_vi_arrows_agree():
    A = zymera.ActionId
    assert KEY_TO_ACTION["w"] == KEY_TO_ACTION["k"] == KEY_TO_ACTION["up"]    == A.NORTH
    assert KEY_TO_ACTION["d"] == KEY_TO_ACTION["l"] == KEY_TO_ACTION["right"] == A.EAST


def test_quit_keys_and_fog_cycle():
    assert _QUIT_KEYS == ("q", "escape")
    assert _FOG_CYCLE == ("off", "self", "all")


def test_disable_matplotlib_keymaps_clears_conflicts():
    _disable_matplotlib_keymaps()
    for keymap in _KEYMAP_BLOCK:
        assert plt.rcParams[keymap] == []


def test_fog_off_no_hidden(small_world):
    mask = compute_fog_mask(small_world, "off", 2, 0)
    assert mask.shape == (small_world.grid_h, small_world.grid_w)
    assert not mask.any()


def test_fog_all_covers_self_modes(small_world):
    """'all' visibility ⊇ each agent's 'self' visibility."""
    mask_all = compute_fog_mask(small_world, "all", 2, 0)
    for i in range(small_world.n_agents):
        mask_self = compute_fog_mask(small_world, "self", 2, i)
        # visible-in-self ⊆ visible-in-all
        assert ((~mask_self) & mask_all).sum() == 0


# ---- rendering helpers -----------------------------------------------------


def test_draw_frame_one_scatter_per_agent():
    w = zymera.World.initial(__import__("jax").random.PRNGKey(0),
                             grid_h=4, grid_w=4, n_agents=3)
    fig, ax = plt.subplots()
    try:
        draw_frame(ax, w, title="three agents")
        assert len(ax.images) == 1            # visited heatmap
        assert len(ax.collections) == 3       # one scatter per agent
        assert ax.get_title() == "three agents"
    finally:
        plt.close(fig)


def test_draw_frame_replaces_artists_on_redraw(small_world):
    fig, ax = plt.subplots()
    try:
        draw_frame(ax, small_world, title="a")
        draw_frame(ax, small_world, title="b")
        assert len(ax.images) == 1
        assert len(ax.collections) == 2
        assert ax.get_title() == "b"
    finally:
        plt.close(fig)


def test_grid_lines_minor_ticks():
    fig, ax = plt.subplots()
    try:
        draw_grid_lines(ax, h=5, w=5)
        assert len(ax.get_xticks(minor=True)) == 6
        assert len(ax.get_yticks(minor=True)) == 6
    finally:
        plt.close(fig)


def test_comm_graph_one_line_per_pair(small_world):
    fig, ax = plt.subplots()
    try:
        adj = np.ones((small_world.n_agents, small_world.n_agents), dtype=bool)
        before = len(ax.lines)
        draw_comm_graph(ax, small_world, adj)
        # N*(N-1)/2 unique pairs
        N = small_world.n_agents
        assert len(ax.lines) - before == N * (N - 1) // 2
    finally:
        plt.close(fig)
