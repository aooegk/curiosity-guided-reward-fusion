import numpy as np


class MovingWindow:

    def __init__(self, window_size=20):
        self.window_size = int(window_size)
        self.values = []

    def add(self, value):
        value = float(value)
        self.values.append(value)

        if len(self.values) > self.window_size:
            self.values.pop(0)

    def mean(self):
        if len(self.values) == 0:
            return None

        return float(np.mean(self.values))

    def std(self):
        if len(self.values) == 0:
            return None

        return float(np.std(self.values))

    def last(self):
        if len(self.values) == 0:
            return None

        return float(self.values[-1])

    def is_ready(self):
        return len(self.values) >= self.window_size

    def state_dict(self):
        return {
            "window_size": self.window_size,
            "values": self.values.copy(),
            "mean": self.mean(),
            "std": self.std(),
            "last": self.last(),
            "is_ready": self.is_ready(),
        }


class BetaScheduler:

    def __init__(
        self,
        beta_min=0.0,
        beta_max=0.5,
        q_loss_window=20,
        q_loss_stable_std=1.0,
        q_loss_unstable_std=5.0,
        abnormal_q_loss=100.0,
    ):

        if beta_min < 0:
            raise ValueError(
                "beta_min must be non-negative."
            )

        if beta_max < beta_min:
            raise ValueError(
                "beta_max must be larger than or equal to beta_min."
            )

        if q_loss_window <= 0:
            raise ValueError(
                "q_loss_window must be positive."
            )

        if q_loss_unstable_std <= q_loss_stable_std:
            raise ValueError(
                "q_loss_unstable_std must be larger than q_loss_stable_std."
            )


        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)


        self.q_loss_window = MovingWindow(
            window_size=q_loss_window
        )

        self.q_loss_stable_std = float(
            q_loss_stable_std
        )

        self.q_loss_unstable_std = float(
            q_loss_unstable_std
        )

        self.abnormal_q_loss = float(
            abnormal_q_loss
        )


        self.last_q_loss = None


        # RND 初始误差
        self.rnd_init_error = None

        self.last_rnd_error = 0.0
        self.last_rnd_score = 1.0


        self.last_stability = 1.0
        self.last_beta = self.beta_max


    def compute_stability(self, q_loss):


        q_loss = float(q_loss)

        self.q_loss_window.add(q_loss)


        if q_loss >= self.abnormal_q_loss:
            return 0.0


        if not self.q_loss_window.is_ready():
            return 1.0


        q_std = self.q_loss_window.std()


        if q_std <= self.q_loss_stable_std:
            return 1.0


        if q_std >= self.q_loss_unstable_std:
            return 0.0


        stability = 1.0 - (
            (q_std - self.q_loss_stable_std)
            /
            (
                self.q_loss_unstable_std
                -
                self.q_loss_stable_std
            )
        )


        stability = np.clip(
            stability,
            0.0,
            1.0
        )


        return float(stability)



    def compute_rnd_score(self, rnd_error):


        rnd_error = float(rnd_error)


        if self.rnd_init_error is None:
            self.rnd_init_error = rnd_error


        rnd_score = (
            rnd_error
            /
            (
                self.rnd_init_error
                +
                1e-8
            )
        )


        rnd_score = np.clip(
            rnd_score,
            0.0,
            1.0
        )


        return float(rnd_score)



    def update(
        self,
        q_loss,
        rnd_error,
    ):


        q_loss = float(q_loss)
        rnd_error = float(rnd_error)


        stability = self.compute_stability(
            q_loss
        )


        rnd_score = self.compute_rnd_score(
            rnd_error
        )


        factor = (
            stability
            *
            rnd_score
        )


        beta = (
            self.beta_min
            +
            (
                self.beta_max
                -
                self.beta_min
            )
            *
            factor
        )


        beta = np.clip(
            beta,
            self.beta_min,
            self.beta_max
        )


        self.last_q_loss = q_loss

        self.last_rnd_error = rnd_error

        self.last_rnd_score = float(
            rnd_score
        )

        self.last_stability = float(
            stability
        )

        self.last_beta = float(
            beta
        )


        return float(beta)



    def get_beta(self):


        return float(
            self.last_beta
        )



    def state_dict(self):


        q_window_state = (
            self.q_loss_window.state_dict()
        )


        return {

            "beta":
                self.last_beta,

            "beta_min":
                self.beta_min,

            "beta_max":
                self.beta_max,


            "stability":
                self.last_stability,


            "rnd_error":
                self.last_rnd_error,

            "rnd_score":
                self.last_rnd_score,


            "q_loss":
                self.last_q_loss,


            "q_loss_mean":
                q_window_state["mean"],

            "q_loss_std":
                q_window_state["std"],

            "q_loss_last":
                q_window_state["last"],

            "q_loss_window_ready":
                q_window_state["is_ready"],
        }



    def print_info(self):


        state = self.state_dict()


        print("=" * 60)

        print("Beta Scheduler")

        print(
            f"beta          : "
            f"{state['beta']:.6f}"
        )

        print(
            f"stability     : "
            f"{state['stability']:.4f}"
        )

        print(
            f"rnd_error     : "
            f"{state['rnd_error']:.6f}"
        )

        print(
            f"rnd_score     : "
            f"{state['rnd_score']:.4f}"
        )

        print(
            f"q_loss        : "
            f"{state['q_loss']}"
        )

        print(
            f"q_loss_mean   : "
            f"{state['q_loss_mean']}"
        )

        print(
            f"q_loss_std    : "
            f"{state['q_loss_std']}"
        )

        print("=" * 60)