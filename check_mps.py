import torch


def main() -> None:
    print(torch.backends.mps.is_available())


if __name__ == "__main__":
    main()
