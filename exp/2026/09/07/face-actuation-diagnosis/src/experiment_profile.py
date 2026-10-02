"""Normal Comet/local experiment evidence without automatic Git mutations."""

from liblaf.cherries import core, plugins, profiles


class ProfileCometNoCommit(profiles.Profile):
    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run
