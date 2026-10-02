"""Comet/local evidence without Git mutations or slow environment inspection."""

import comet_ml
from liblaf.cherries import core, plugins, profiles


class Comet(plugins.Comet):
    @core.impl
    def start(self) -> None:
        experiment = comet_ml.start(
            project_name=self.run.project_name,
            experiment_config=comet_ml.ExperimentConfig(
                disabled=False,
                name=self.run.run_name,
                tags=self.run.tags,
                log_env_details=False,
                log_git_patch=False,
                auto_log_co2=False,
            ),
        )
        self.run.log_other("cherries/comet/url", experiment.url)


class ProfileCometNoCommit(profiles.Profile):
    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(Comet(run=run))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run
