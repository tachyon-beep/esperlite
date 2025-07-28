"""Quick test for simplified Triton kernel."""

import torch

from esper.morphogenetic_v2.triton.simple_forward_kernel import SimpleTritonLayer


def test_simple_kernel():
    """Test the simplified kernel works."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = torch.device('cuda:0')

    # Create layer
    layer = SimpleTritonLayer(
        hidden_dim=512,
        num_seeds=100,
        chunk_size=64
    ).to(device)

    # Test 1: Identity for dormant seeds
    x = torch.randn(8, 512, device=device)
    y = layer(x)

    print("Test 1 - Identity transform:")
    print(f"  Input/output match: {torch.allclose(x, y)}")
    print(f"  Max difference: {(x - y).abs().max().item():.6f}")

    # Test 2: Active seeds
    layer.activate_seed(0, blueprint_id=0, strategy=1)  # Additive
    layer.activate_seed(1, blueprint_id=1, strategy=0)  # Multiplicative

    y2 = layer(x)

    print("\nTest 2 - Active seeds:")
    print(f"  Output changed: {not torch.allclose(x, y2)}")

    # Check specific chunks
    chunk0 = y2[:, :64]
    chunk1 = y2[:, 64:128]
    chunk2 = y2[:, 128:192]

    input0 = x[:, :64]
    input1 = x[:, 64:128]
    input2 = x[:, 128:192]

    print(f"  Chunk 0 modified: {not torch.allclose(chunk0, input0)}")
    print(f"  Chunk 1 modified: {not torch.allclose(chunk1, input1)}")
    print(f"  Chunk 2 unchanged: {torch.allclose(chunk2, input2)}")

    # Test 3: Performance
    import time

    # Warmup
    for _ in range(10):
        _ = layer(x)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(100):
        _ = layer(x)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print("\nTest 3 - Performance:")
    print(f"  Time per forward: {elapsed/100*1000:.2f} ms")
    print(f"  Throughput: {8*512*4*2*100/elapsed/1e9:.2f} GB/s")


if __name__ == '__main__':
    test_simple_kernel()
