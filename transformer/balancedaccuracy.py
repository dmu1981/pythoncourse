class BalancedAccuracy():
  def __init__(self, ignore_tokens=None):
    self.ignore_tokens = ignore_tokens
    self.reset()

  def get(self):
    dct = { token: 0 for token in self.token_total.keys() }

    for token in self.token_correct.keys():
      dct[token] = self.token_correct[token] / self.token_total[token]

    bacc = 0
    for token in self.token_total.keys():
      bacc += dct[token]
      
    return dct, bacc / len(self.token_total.keys()), self.correct / self.total
  
  def reset(self):
    self.total = 0
    self.correct = 0

    self.token_correct = {}
    self.token_total = {}

  def update(self, predicted_tokens, correct_tokens):
    for predicted_token, correct_token in zip(predicted_tokens, correct_tokens):
      if correct_token in self.ignore_tokens:
        continue

      self.total += 1
      if correct_token in self.token_total:
        self.token_total[correct_token] += 1
      else:
        self.token_total[correct_token] = 1

      if predicted_token == correct_token:
        self.correct += 1
        if correct_token in self.token_correct:
          self.token_correct[correct_token] += 1
        else:
          self.token_correct[correct_token] = 1