SHIONNEpisodeActive <- false;
SHIONNEpisodeFinished <- false;

function SHIONNEmitEvent(eventName, value)
{
    printl("EVT|" + eventName + "|" + value);
}

function SignalChamberReady()
{
    if (SHIONNEpisodeActive)
    {
        return;
    }

    SHIONNEpisodeActive = true;
    SHIONNEpisodeFinished = false;
    SHIONNEmitEvent("chamber_ready", Time());
}

function SignalGoalReached()
{
    if (!SHIONNEpisodeActive || SHIONNEpisodeFinished)
    {
        return;
    }

    SHIONNEpisodeActive = false;
    SHIONNEpisodeFinished = true;
    SHIONNEmitEvent("goal_reached", 1);
}

function SignalEpisodeFailed(reason)
{
    if (!SHIONNEpisodeActive || SHIONNEpisodeFinished)
    {
        return;
    }

    SHIONNEpisodeActive = false;
    SHIONNEpisodeFinished = true;
    SHIONNEmitEvent("episode_failed", reason);
}

function SignalTimeout()
{
    SignalEpisodeFailed("timeout");
}

function SignalOutOfBounds()
{
    SignalEpisodeFailed("out_of_bounds");
}

function SignalPlayerDeath()
{
    SignalEpisodeFailed("player_death");
}

function ResetEpisodeSignals()
{
    SHIONNEpisodeActive = false;
    SHIONNEpisodeFinished = false;
}
