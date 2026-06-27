from inspect import signature

from gym.envs.registration import register

register_kwargs = {}
if 'disable_env_checker' in signature(register).parameters:
    register_kwargs['disable_env_checker'] = True

register(
    id='OvercookedMultiEnv-v0',
    entry_point='overcookedgym.overcooked:OvercookedMultiEnv',
    **register_kwargs
)
