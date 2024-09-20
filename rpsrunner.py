try:
    from multiprocessing import Pool, cpu_count
except ImportError:
    Pool = None

import getopt
import glob
import os
import random
import sys
import time
import traceback

########

VERSION = "1.0.1"
MATCHES = 10
POOL_SIZE = 1

if Pool is not None:
    try:
        POOL_SIZE = cpu_count()
    except Exception:
        pass

WINDOWS = False
try:
    WINDOWS = sys.platform.startswith("win")
except Exception:
    WINDOWS = False

########

class Bot:
    """Basic bot class to wrap bot functions"""
    def __init__(self, name, code=None):
        """
        name should be a unique identifier and must be a readable
        filename if code is not specified
        """
        self.name = name
        if code is None:
            self.load_code()
        else:
            self.code = code

        self.reset()

    def __eq__(self, other):
        return self.name == other.name

    def get_move(self, input):
        """Get the next move for the bot given input
        input must be "R", "P", "S" or ""
        """
        if self._code is None:
            self.compile_code()

        self.scope["input"] = input
        exec(self._code, self.scope)
        self.output = self.scope["output"]
        return self.output

    def compile_code(self):
        self._code = compile(self.code, '<string>', 'exec')

    def reset(self):
        """Resets bot for another round. This must be called before trying
        to pass the bot between workers, or you may see obscure errors from failures
        to pickle the bots scope dictionary."""
        self.scope = dict()

        # this will hold compiled code, but it apparently can't be
        # pickled? so we'll have to do it later. XXX check into this
        self._code = None

    def load_code(self):
        """Load bot code from the file specified by the name attribute"""
        with open(self.name, "r") as f:
            self.code = f.read()

# used to evaluate a pair of moves
# scoring[bot1_move][bot2_move]
# 1 = bot1 won, -1 = bot2 won, 0 = tie
# TODO: move into ContestResult?
scoring = {
    "R": {"R": 0, "P": -1, "S": 1},
    "P": {"R": 1, "P": 0, "S": -1},
    "S": {"R": -1, "P": 1, "S": 0}
}

class ContestResult:
    """Used to track and report on the status of a contest. Shared values
    are reported from the perspective of bot1. For example, score > 0 indicates
    that bot1 won by that many points. score < 0 indicates bot 1 lost by that
    many points."""
    # TODO bot-specific data should be a separate object. ContestResult
    # should track two of these objects and move most of the bot-specific
    # data below into them.
    def __init__(self, bot1, bot2):
        self.bot1 = bot1
        self.bot2 = bot2
        self.bot1_disqualified = False
        self.bot2_disqualified = False
        self.finalized = False
        self.errors = False
        self.error_string = ""

        self.wins1 = 0
        self.wins2 = 0
        self.ties1 = 0
        self.ties2 = 0
        self.losses1 = 0
        self.losses2 = 0
        self.score = 0
        self.played = 0
        self.history1 = []
        self.history2 = []
        self.score_history = []
        self.start_time = None
        self.end_time = None
        self.run_time = 0.0
        self.winner = None
        self.loser = None

    def start(self):
        self.start_time = time.time()

    def score_moves(self, move1, move2):
        """This function is called to score and track each pair of moves
        from a contest."""

        score = 0
        try:
            score = scoring[move1][move2]
        except KeyError:
            # TODO disqualify bot and exit contest
            if move1 not in "RPS":
                score = -1
            elif move2 not in "RPS":
                score = 1
            else:
                raise Exception("Can't score %s and %s?!" % (move1, move2))

        if score > 0:
            self.wins1 += 1
            self.losses2 += 1
        elif score < 0:
            self.losses1 += 1
            self.wins2 += 1
        else:
            self.ties1 += 1
            self.ties2 += 1

        self.score += score
        self.history1.append(move1)
        self.history2.append(move2)
        self.score_history.append(score)
        self.played += 1

        return score

    def finalize(self, errors=False, error_string=""):
        """Called once a contest is complete to do some final bookkeeping.
        This is REQUIRED if multiprocessing features are in use."""
        # the bots must be reset before being passed between workers
        # see comments under Bot.reset()
        self.bot1.reset()
        self.bot2.reset()

        self.errors = errors
        self.error_string = error_string
        self.history1 = "".join(self.history1)
        self.history2 = "".join(self.history2)
        self.end_time = time.time()
        self.run_time = self.end_time - self.start_time

        if self.wins1 > self.wins2:
            self.winner = self.bot1
            self.loser = self.bot2
        elif self.wins1 < self.wins2:
            self.winner = self.bot2
            self.loser = self.bot1

        self.finalized = True

    def __str__(self):
        game = "%s vs %s:" % (self.bot1.name, self.bot2.name)
        if self.bot1_disqualified:
            return "%s bot 1 disqualified" % game
        elif self.bot2_disqualified:
            return "%s bot 2 disqualified" % game
        elif self.finalized:
            return "%s score %d, took %.2f seconds" % \
                   (game, self.score, self.run_time)
        else:
            return "%s score %d -- not final" % (game, self.score)


class Contest:
    """Contest object handles running a contest between two sets of bots."""
    def __init__(self, bot1, bot2, rounds=1000):
        self.bot1 = bot1
        self.bot2 = bot2
        self.rounds = rounds
        self.result = ContestResult(bot1, bot2)

        # isolate random number generator
        r1 = random.random()
        r2 = random.random()
        base_rng = random.getstate()

        random.seed(r1)
        self.bot1_rng = random.getstate()

        random.seed(r2)
        self.bot2_rng = random.getstate()

        random.setstate(base_rng)

    def run(self):
        """Runs the configured contest and reports a ContestResult"""
        self.result.start()

        base_rng = random.getstate()
        input1 = input2 = output1 = output2 = ""
        errors = False
        error_string = ""
        for i in range(self.rounds):
            random.setstate(self.bot1_rng)
            try:
                output1 = self.bot1.get_move(input1)
            except KeyboardInterrupt:
                raise
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                exc_string = "".join(traceback.format_exception(exc_type,
                                                                exc_value, exc_traceback))
                error_string = "Error from %s\n%s" % (self.bot1.name,
                                                      exc_string)
                errors = True
                self.result.bot1_disqualified = True
            else:
                if output1 not in "RPS":
                    errors = True
                    self.result.bot1_disqualified = True
                    error_string = "bot1 did not make a valid move"
            self.bot1_rng = random.getstate()

            random.setstate(self.bot2_rng)
            try:
                output2 = self.bot2.get_move(input2)
            except KeyboardInterrupt:
                raise
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                exc_string = "".join(traceback.format_exception(exc_type,
                                                                exc_value, exc_traceback))
                error_string = "Error from %s\n%s" % (self.bot2.name,
                                                      exc_string)
                errors = True
                self.result.bot2_disqualified = True
            else:
                if output2 not in "RPS":
                    errors = True
                    self.result.bot2_disqualified = True
                    error_string = "bot2 did not make a valid move"
            self.bot2_rng = random.getstate()

            if errors:
                break

            self.result.score_moves(output1, output2)
            input1 = output2
            input2 = output1

            # TODO add early bail out like official contest

        self.result.finalize(errors=errors, error_string=error_string)
        return self.result

########

def usage():
    print("usage: %s [-h] [-c contests] [-m matches] [-p pool_size]" % sys.argv[0])
    sys.exit()

def pool_init():
    """Called by each worker process on startup. Used to install a signal
    handler."""
    try:
        import signal
    except ImportError:
        return

    def sigint_handler(signal, frame):
        sys.exit()

    signal.signal(signal.SIGINT, sigint_handler)


def main():
    # parse options
    global MATCHES, POOL_SIZE
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hc:m:p:")
    except getopt.GetoptError as err:
        print(err)
        usage()

    contests = 1
    for o, a in opts:
        if o == "-h":
            usage()
        elif o == "-c":
            contests = int(a)
        elif o == "-m":
            MATCHES = int(a)
        elif o == "-p":
            POOL_SIZE = int(a)

    if len(args) < 1:
        usage()

    try:
        bots = []
        for arg in args:
            bots.extend(glob.glob(arg))

        print("Found %d bots" % len(bots))
        sys.stdout.flush()

        workers = Pool(POOL_SIZE, pool_init)
        tasks = []

        for i in range(contests):
            for b1 in range(len(bots)):
                for b2 in range(b1 + 1, len(bots)):
                    bot1 = Bot(bots[b1])
                    bot2 = Bot(bots[b2])

                    for match in range(MATCHES):
                        contest = Contest(bot1, bot2)
                        tasks.append(contest)

        results = workers.map(Contest.run, tasks)
        workers.close()
        workers.join()

        for result in results:
            print(result)

    except KeyboardInterrupt:
        print("Aborting contests")
        sys.exit()

if __name__ == "__main__":
    main()
