from .save_served import save_served

def decide_candy(name, served_set):
    if name in served_set:
        return {
            "dispense": False,
            "message": f"{name} → CANDY NOT DISPENSED"
        }

    served_set.add(name)
    save_served(served_set)

    return {
        "dispense": True,
        "message": f"{name} → DISPENSED CANDY"
    }