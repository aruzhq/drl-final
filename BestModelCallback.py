from stable_baselines3.common.callbacks import BaseCallback

class BestModelCallback(BaseCallback):
    def __init__(self, check_freq, save_path, metric_name, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.metric_name = metric_name
        self.best_value = -1e9

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            logs = self.model.logger.name_to_value
            if self.metric_name in logs:
                current_value = logs[self.metric_name]
                if current_value > self.best_value:
                    self.best_value = current_value
                    self.model.save(self.save_path)
        return True