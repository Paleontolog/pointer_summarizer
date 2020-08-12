import torch.utils.tensorboard.summary as summary

def calc_running_avg_loss(loss, running_avg_loss, decay=0.99):
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)
  return running_avg_loss


def write_summary(tag, running_avg_loss, summary_writer, step, decay=0.99):
  loss_sum = summary.Summary()
  tag_name = f'%s/decay=%f' % (tag, decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)


def calc_and_write_running_avg_loss(loss, tag, running_avg_loss, summary_writer, step, decay=0.99):
  running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, decay)
  write_summary(tag, running_avg_loss, summary_writer, step, decay)
  return running_avg_loss