// Optional goal_choice instrumentation. Load AFTER shionn_events.nut on the
// same logic_script. Evaluator-only events; never policy inference features.
function SignalWaypointReached(waypointId)
{
    if (!SHIONNEpisodeActive || SHIONNEpisodeFinished)
    {
        return;
    }
    if (waypointId != "A" && waypointId != "B")
    {
        return;
    }
    SHIONNEmitEvent("waypoint_reached", waypointId);
}

// Hammer entity output parameters do not reliably preserve embedded quotes.
function SignalWaypointA()
{
    SignalWaypointReached("A");
}

function SignalWaypointB()
{
    SignalWaypointReached("B");
}
