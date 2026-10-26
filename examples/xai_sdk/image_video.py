"""Trace xAI image and video generation, including the exact response cost."""

from xai_sdk import Client

import genai_otel


def main():
    genai_otel.instrument()
    client = Client()

    # Billable API request. The SDK response exposes xAI's exact `cost_usd`.
    image = client.image.sample(
        model="grok-imagine-image-2.0",
        prompt="A small red fox reading beside a rainy window, storybook style",
    )
    print("Image cost: $%s" % image.cost_usd)

    # Video generation is also billable. Uncomment to try it.
    # video = client.video.generate(
    #     model="grok-imagine-video-1.5",
    #     prompt="A paper boat drifting down a quiet stream",
    #     duration=5,
    # )
    # print("Video cost: $%s" % video.cost_usd)


if __name__ == "__main__":
    main()
