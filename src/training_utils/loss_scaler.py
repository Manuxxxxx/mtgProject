class LossScaler:
    def __init__(self, smoothing=0.95, epsilon=1e-8, initial_scale=1.0):
        self.avg_tag_loss = None
        self.avg_synergy_loss = None
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.initial_scale = initial_scale

    def update(self, tag_loss_val, synergy_loss_val):
        if self.avg_tag_loss is None:
            self.avg_tag_loss = tag_loss_val
            self.avg_synergy_loss = synergy_loss_val
        else:
            self.avg_tag_loss = self.smoothing * self.avg_tag_loss + (1 - self.smoothing) * tag_loss_val
            self.avg_synergy_loss = self.smoothing * self.avg_synergy_loss + (1 - self.smoothing) * synergy_loss_val

    def get_scaled_total_loss(self, tag_loss, synergy_loss, alpha=1.0):
        self.update(tag_loss.item(), synergy_loss.item())
        # Compute scale factor to bring tag loss to same level as synergy loss
        if self.avg_tag_loss < self.epsilon:
            scale_factor = self.initial_scale
        else:
            scale_factor = self.avg_synergy_loss / (self.avg_tag_loss + self.epsilon)

        scaled_tag_loss = scale_factor * tag_loss
        return synergy_loss + alpha * scaled_tag_loss, scale_factor

