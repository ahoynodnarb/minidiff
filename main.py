def main():
    from utils import print_graph
    import networkx as nx
    import matplotlib.pyplot as plt
    from tensor import Tensor
    
    b = Tensor([[0, 2, -2, 1], [-1, -1, -2, -2]])
    t1 = Tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    t2 = Tensor([[0, 1, -1, 2], [1, 0, 1, 0]])
    t3 = t1 + t2 - b
    t4 = t3 * t2
    t5 = 2 * t4 - t1*t3
    # t5 = 2 * t4 - t1*t3
    # t5 = 2 * t3 * t2 - t1 * (t1 + t2 - b)
    # t5 = 2 * (t1 + t2 - b) * t2 - t1^2 - t1 * t2 + t1 * b
    # t5 = 2 * t1 * t2 + 2 * t2^2 - 2 * t2 * b - t1^2 - t1 * t2 + t1 * b
    # t5 = t1 * t2 + 2 * t2^2 - t1^2 - 2 * t2 * b + t1 * b
    # dt5/dt4 = 2
    # dt5/dt3 = 2 * t2 - t1
    # dt5/dt2 = t1 + 4 * t2 - 2 * b
    # dt5/dt1 = t2 - 2 * t1 + b
    # dt5/db = - 2 * t2 + t1
    t5.backward()

    print(f"{b.grad=}")
    print(f"{(-2 * t2 + t1)=}")
    print(f"{t1.grad=}")
    print(f"{(t2 - 2 * t1 + b)=}")
    print()
    print(f"{t2.grad=}")
    print(f"{(t1 + 4 * t2 - 2 * b)=}")
    print()
    print(f"{t3.grad=}")
    print(f"{(2 * t2 - t1)=}")
    print()
    print(f"{t4.grad=}")
    print(f"{2=}")
    print()
    print(f"{t5.grad=}")
    print(f"{1=}")

if __name__ == "__main__":
    main()