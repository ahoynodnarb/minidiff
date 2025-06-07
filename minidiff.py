from contextvars import ContextVar  

allow_grad = True

class no_grad:
    def __enter__(self):
        set_allow_grad(False)
        
    def __exit__(self, type, value, traceback):
        set_allow_grad(True)
    
def set_allow_grad(allow):
    global allow_grad
    allow_grad = allow
        
def grad_allowed():
    global allow_grad
    return allow_grad

if __name__ == "__main__":
    with no_grad():
        print(grad_allowed())
    print(grad_allowed())