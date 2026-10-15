async def comfy_entrypoint():
    from .nodes import Flux3PEExtension
    return Flux3PEExtension()
