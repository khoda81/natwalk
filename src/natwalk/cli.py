"""Shared command-line surface for Natwalk example applications."""

from __future__ import annotations

import argparse


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show concrete defaults while leaving semantic ``None`` defaults to prose."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None:
            return action.help or ""
        return super()._get_help_string(action)


def add_tui_arguments(
    parser: argparse.ArgumentParser,
    *,
    max_tokens_default: int,
    max_tokens_help: str,
) -> None:
    """Add the shared Natwalk TUI options to one application parser."""
    natwalk = parser.add_argument_group("natwalk")
    natwalk.add_argument(
        "--max-tokens",
        type=int,
        default=max_tokens_default,
        help=max_tokens_help,
    )
    natwalk.add_argument(
        "--tree-lines",
        type=int,
        help="maximum rendered tree rows; default uses the remaining terminal height",
    )
    natwalk.add_argument(
        "--budget-nats",
        type=float,
        default=1.5,
        help="maximum cumulative surprisal of the highlighted/accepted suggestion, in nats",
    )
    natwalk.add_argument(
        "--budget-step",
        type=float,
        default=0.25,
        help="amount '[' and ']' change --budget-nats by",
    )
    natwalk.add_argument(
        "--max-tree-bytes",
        type=int,
        help=(
            "soft limit on retained authoritative tree-distribution bytes; at the limit "
            "autonomous search pauses but explicit navigation still works; default unlimited"
        ),
    )
