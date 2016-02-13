#!/usr/bin/env python3

import argparse
import asyncio
import asyncio.futures
import concurrent.futures
import logging
import math
import random
import sys

log = logging.getLogger(__name__)

cat = """

  |\      _,,,---,,_
  /,`.-'`'    -.  ;-;;,_
 |,4-  ) )-,_..;\ (  `'-'
'---''(_/--'  `-'\_)

"""


@asyncio.coroutine
def timed_input(prompt, timeout=0):
    """Wait for input from the user

    Note that user input is not available until the user pressed the enter or
    return key.

    Arguments:
        prompt  - text that is used to prompt the users response
        timeout - the number of seconds to wait for input

    Raises:
        An asyncio.futures.TimeoutError is raised if the user does not provide
        input within the specified timeout period.

    Returns:
        A string containing the users response

    """
    # Write the prompt to stdout
    sys.stdout.write(prompt)
    sys.stdout.flush()

    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    # The response callback will receive the users input and put it onto the
    # queue in an independent task.
    def response():
        loop.create_task(queue.put(sys.stdin.readline()))

    # Create a reader so that the response callback is invoked if the user
    # provides input on stdin.
    loop.add_reader(sys.stdin.fileno(), response)

    try:
        # Wait for an item to be placed on the queue. The only way this can
        # happen is if the reader callback is invoked.
        return (yield from asyncio.wait_for(queue.get(), timeout=timeout))

    finally:
        # Whatever happens, we should remove the reader so that the callback
        # never gets called.
        loop.remove_reader(sys.stdin.fileno())


def binomial(p, k, n):
    u = k * math.log(p)
    v = (n - k) * math.log(1 - p)

    a = sum(math.log(_) for _ in range(1, n + 1))
    b = sum(math.log(_) for _ in range(1, k + 1))
    c = sum(math.log(_) for _ in range(1, n - k + 1))
    w = a - b - c

    return math.exp(u + v + w)


class Models(object):
    def __init__(self):
        self.px0 = 1
        self.py0 = 7
        self.pz0 = 1

    def level(self, number):
        px = binomial(0.9, number.correct, number.total)
        py = binomial(0.5, number.correct, number.total)
        pz = binomial(0.3, number.correct, number.total)

        x = px * self.px0
        y = py * self.py0
        z = pz * self.pz0

        px = x / (x + y + z)
        py = y / (x + y + z)
        pz = z / (x + y + z)

        maxp = max(px, py, pz)

        if px == maxp:
            return 'mastery'

        if py == maxp:
            return 'learning'

        return 'confused'



class Result(object):
    def __init__(self, number):
        self.number = number
        self.correct = 0
        self.total = 0

    def __repr__(self):
        return "{}: {}/{}".format(self.number, self.correct, self.total)


class Question(object):
    def __init__(self, tester, x, y):
        self.x = x
        self.y = y
        self.tester = tester

    def answer(self, z):
        if z == self.x + self.y:
            self.tester.correct(self.x, self.y)
        else:
            self.tester.incorrect(self.x, self.y)


class Tester(object):
    def __init__(self):
        self.results = {n:Result(n) for n in range(10)}
        self.state = {n:'learning' for n in self.results}

    @property
    def confused(self):
        return [n for n in self.results if self.state[n] == 'confused']

    @property
    def learning(self):
        return [n for n in self.results if self.state[n] == 'learning']

    @property
    def mastery(self):
        return [n for n in self.results if self.state[n] == 'mastery']

    def question(self):
        x = random.choice(self.learning + self.confused)
        y = random.choice(self.learning + self.confused + self.mastery)

        return Question(self, x, y)

    def correct(self, x, y):
        self.results[x].correct += 1
        self.results[y].correct += 1
        self.results[x].total += 1
        self.results[y].total += 1

    def incorrect(self, x, y):
        self.results[x].total += 1
        self.results[y].total += 1

    def update_state(self):
        models = Models()

        for n, r in self.results.items():
            state = models.level(r)

            if state == self.state[n]:
                continue

            msg = "transition from '{}' to '{}' for {}"
            print(msg.format(self.state[n], state, n))

            self.state[n] = state


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--summary', '-s', action='store_true')
    parser.add_argument('--timeout', '-t', default=10, type=int)

    args = parser.parse_args(argv)

    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    t = Tester()
    loop = asyncio.get_event_loop()

    try:
        while True:
            try:
                @asyncio.coroutine
                def ask_question():
                    nonlocal t
                    q = t.question()

                    try:
                        coro = timed_input('{} + {} = '.format(q.x, q.y), args.timeout)
                        a = yield from coro

                        q.answer(int(a))

                    except ValueError:
                        t.incorrect(q.x, q.y)

                    except asyncio.futures.TimeoutError:
                        print('X')
                        t.incorrect(q.x, q.y)

                    t.update_state()

                loop.run_until_complete(ask_question())

            except Exception as e:
                log.exception(e)

            if not (t.learning + t.confused):
                print("congratulations! You have mastered the numbers")
                print(", ".join(map(str, t.mastery)))
                break

    except (SystemExit, KeyboardInterrupt):
        # At this point the event loop as been stopped. To clean up cancel all
        # of the tasks and then allow the event loop to run again.
        tasks = asyncio.Task.all_tasks()
        if tasks:
            for task in tasks:
                task.cancel()

            loop.run_forever()

    finally:
        if loop.is_running():
            loop.close()

        if args.summary:
            print('\n\nSUMMARY\n' + 50 * '=' + '\n')

            px0 = 1
            py0 = 7
            pz0 = 1

            for n in t.results.values():
                px = binomial(0.9, n.correct, n.total)
                py = binomial(0.5, n.correct, n.total)
                pz = binomial(0.3, n.correct, n.total)

                x = px * px0
                y = py * py0
                z = pz * pz0

                px = x / (x + y + z)
                py = y / (x + y + z)
                pz = z / (x + y + z)

                if px == max(px, py, pz):
                    print("{}: {}|{}".format(n, 'mastery', px))
                elif py == max(px, py, pz):
                    print("{}: {}|{}".format(n, 'learning', py))
                elif pz == max(px, py, pz):
                    print("{}: {}|{}".format(n, 'confused', pz))

    print('\n\nGood job! Take a break.')
    print(cat)


if __name__ == "__main__":
    logging.basicConfig()
    main()
