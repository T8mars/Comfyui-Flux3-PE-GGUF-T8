"""Check the installed wheel's native PE sampler without downloading a model."""
import ctypes
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flux3_pe.protocol import prepare_request
from flux3_pe.wheel_backend import generated_sampler, load_wheel, wheel_identity


def main():
    wheel_identity()
    api, mtmd_cpp = load_wheel()

    assert callable(mtmd_cpp.mtmd_helper_eval_chunk_single)
    sampler = generated_sampler(type("Vocabulary", (), {"n_vocab": lambda self: 3})(), prepare_request("A ball.").payload())
    try:
        def logits():
            values = (api.llama_token_data * 3)(*[api.llama_token_data(i, value, 0) for i, value in enumerate((3.0, 2.5, 2.0))])
            array = api.llama_token_data_array(values, 3, -1, False)
            api.llama_sampler_apply(sampler.sampler, ctypes.byref(array))
            return [round(value.logit, 4) for value in values]
        # Prompt tokens were never accepted into this chain.
        assert logits() == [3.0, 2.5, 2.0]
        api.llama_sampler_accept(sampler.sampler, 0)
        assert logits() == [1.5, 2.5, 2.0]
        print(json.dumps({"wheel_version": api.__version__, "mtmd_available": True, "generated_only_penalties": True, "server_executable_used": False}))
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
