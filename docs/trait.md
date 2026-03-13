Spec Name
Type Marker Only Control
Goal
Test whether merely exposing agent type labels in observation, without modifying reward or environment rules, is sufficient to induce type-based targeting behavior.
New Config Flags
--agent-types-mode {off,observe}
--agent-type-layout {none,3circle_1triangle,all_circle}
--triangle-agent-id INT (optional)
Activation Condition
This control condition is active only when:
num_robots = 4
agent-types-mode = observe
agent-type-layout = 3circle_1triangle
Type Assignment
3 agents are assigned type circle
1 agent is assigned type triangle
Assignment may be fixed by triangle-agent-id or randomized per episode
Observation Change
When agent-types-mode=observe:
add self_type to self features
add other_type to each other-agent feature block
When agent-types-mode=off:
observation format remains identical to the original implementation, or semantically identical if a padded implementation is used
Environment Dynamics
No change.
Reward Function
No change.
Training Logic
No change except for observation input dimensionality.
Logging
Replay / evaluation logs must record each agent’s assigned type for post-hoc analysis.
Expected Outcome
No robust or consistent type-based bullying should emerge across seeds if type information is the only added signal.