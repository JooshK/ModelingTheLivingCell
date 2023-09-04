def every_five(n: int):
    """
    Prints integers from 1 to n (inclusive) with every fifth integer represented as a multiplication with five
    :param n: the maximum to count to
    """
    for i in range(1, n + 1):  # loop over numbers from 1 to n
        if i % 5 == 0:  # check for multiples of five
            print("5*" + str(i / 5))
        else:
            print(i)


if __name__ == "__main__":
    every_five(20)
