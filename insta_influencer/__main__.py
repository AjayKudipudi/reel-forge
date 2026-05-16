"""`python -m insta_influencer ...` entry point — Click group."""
from __future__ import annotations

import click

from .cli.cancel import cancel_cmd
from .cli.cleanup import cleanup_cmd
from .cli.generate import generate_cmd
from .cli.logs import logs_cmd
from .cli.post import post_cmd
from .cli.prepare import prepare_cmd
from .cli.retry import retry_cmd
from .cli.stats import stats_cmd
from .cli.status import status_cmd


@click.group()
@click.version_option()
def main() -> None:
    """insta — automated Reels generation."""


main.add_command(prepare_cmd, name="prepare")
main.add_command(generate_cmd, name="generate")
main.add_command(status_cmd, name="status")
main.add_command(logs_cmd, name="logs")
main.add_command(retry_cmd, name="retry")
main.add_command(cancel_cmd, name="cancel")
main.add_command(post_cmd, name="post")
main.add_command(cleanup_cmd, name="cleanup")
main.add_command(stats_cmd, name="stats")


if __name__ == "__main__":
    main()
