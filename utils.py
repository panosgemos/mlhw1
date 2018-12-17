""" Module containing utility functions and classes """


def split(alist, idx):
    """
    Splits a list to the given index.

    :param alist: list to be split
    :param idx: index where the list will be split

    :return: the split list
    """

    return alist[:idx], alist[idx:]


def step_split(alist, step):
    """
    Splits a list into pieces, by n steps.

    :param alist: list to be split
    :param step: number of steps

    :return: the split list
    """

    return [alist[i:i + step] for i in range(0, len(alist), step)]


def partition(alist, n):
    """
    Splits a list into n partitions.

    Algorithm Logic:
    The algorithm works as follows. The given list is separated into to two
    parts, its head and its tail. Both will be used to store the partitions of
    the original list. The head contains partitions which are bigger
    by one element than the tail's partitions. This is in order to solve the
    problem where the list size is not divisible by the partition number.
    Therefore some partitions will have to be smaller that the others. This
    head and tail separation also guarantees that the elements will be
    uniformly distributed. That is only one pair of consecutive partitions will
    differ in size, and the difference will be at most one. These are the last
    partition of the list's head and the first partition of the list's tail.

    First the size of the tail is calculated, which is equal to the
    quotient of the division of the size of the list over the number of
    partitions. As previously noted, the size of the head partitions is
    bigger that the tail partitions by one. Then we calculate the remainder
    of the above division, which value denotes the number of head partitions.

    :author: Panos Gemos

    :param alist: list to be split
    :param n: number of partitions

    :return: the list of partitions of the original list
    """

    # if the list is empty ...
    if len(alist) == 0:
        return alist    # ... return that empty list

    rest = len(alist) % n  # remainder of the list size over the partition size
    part_size = len(alist) // n

    # if the list can be divided to partitions of equal size ...
    if rest == 0:
        # ... split the list by that size and return the resulting partitions
        return step_split(alist, part_size)

    # Calculation of the sizes of the head and tail
    tail_part_size = part_size
    head_part_size = tail_part_size + 1

    # split the list into its head and tail
    head, tail = split(alist, head_part_size*rest)

    # split each head and tail, to extract the final partitions
    head = step_split(head, head_part_size)
    tail = step_split(tail, tail_part_size)

    head.extend(tail)  # merge the tail with the head

    return head


def sort_together(*lists, reverse=False):
    """
    Sorts the given lists, based on the first one. The lists given as input
    are not tampered and the lists returned contain shallow copies of the
    original lists' elements.

    :param lists: lists to be sorted
    :param reverse: False if ascending order True if descending order

    :return: a tuple containing the sorted lists
    """

    # Create the initially empty lists to later store the sorted items
    sorted_lists = tuple([] for _ in range(len(lists)))

    # Unpack the lists, sort them, zip them and iterate over them
    list_zip = zip(*lists)
    aslist = list(list_zip)
    sorted_list_zip = sorted(zip(*lists), reverse = reverse)
    for t in sorted_list_zip:
        # list items are now sorted based on the first list
        for i, item in enumerate(t):    # for each item...
            sorted_lists[i].append(item)  # ...store it in the appropriate list

    return sorted_lists
