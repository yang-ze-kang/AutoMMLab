from mmengine.config.config import ConfigDict
import matplotlib.pyplot as plt
from collections import Counter



def flush_config(cfg,k,value):
    for key in cfg:
        try:
            if key==k:
                cfg[key] = value
            elif isinstance(cfg[key],ConfigDict):
                flush_config(cfg[key],k,value)
        except:
            continue
    return cfg

def array_to_markdown_table(array,id=-1):
    col_widths = [max(len(str(item)) for item in col) for col in zip(*array)]
    header = "| " + " | ".join("{:<{width}}".format(str(item), width=col_width) for item, col_width in zip(array[0], col_widths)) + " |"
    separator = "| " + " | ".join("-" * col_width for col_width in col_widths) + " |"
    rows = []
    for row in array[1:]:
        if id!=-1 and row==id:
            row_str = "| " + " | ".join("{:<{width}}".format(f"**{str(item)}**", width=col_width) for item, col_width in zip(row, col_widths)) + " |"
        else:
            row_str = "| " + " | ".join("{:<{width}}".format(str(item), width=col_width) for item, col_width in zip(row, col_widths)) + " |"
        rows.append(row_str)
    markdown_table = "\n".join([header, separator] + rows)
    return markdown_table

def draw_bar_chart(title,data):
    plt.figure()
    counter = Counter(data)
    values = list(counter.keys())
    counts = list(counter.values())
    print(title, values,counts)

    plt.bar(values, counts)

    plt.title(title)
    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.savefig(f"summary_{title}.png")