import numpy as np
import matplotlib.pyplot as plt

#first need to change text input to real func
def real_change(user_input : str):
    allowed = {} 
    allowed["np"] = np
    allowed["sin"] = np.sin
    allowed["cos"] = np.cos
    allowed["exp"] = np.exp
    allowed["pi"] = np.pi
    allowed["u"] = u

    def f(t):
        allowed["t"] = t
        return eval(user_input, {}, allowed)
    return f

#u is special, need to define seperately, not in numpy
#check type cause numpy is array
def u(t) :
    if isinstance(t, (int,float)):
        if t>=0:
            return 1
        else:
            return 0
    else:
        t=np.array(t)
        return (t>=0).astype(float)
    
#in conv, x(t)*h(t) must not be 0, so need to check non-zero region = support
def support_estimation(f, grid, z_stan=1e-6) : 
    y_val = f(grid) #all the y_val
    support_zone = np.abs(y_val) > z_stan #not 0 then 

    if np.any(support_zone):

        xx=np.flatnonzero(support_zone) #getting support_zone true array 
        a = grid[xx[0]]
        b = grid[xx[-1]]
        return float(a), float(b)
    
    else: 
        return None

#float error due to integration -> pick one dot among similars 
def pick_one(possible_dots, tol=1e-12):
    possible_dots = sorted(possible_dots)
    dot_list = []

    for i in possible_dots:
        if len(dot_list) == 0:
            dot_list.append(i)
            #comparing after filled first one
        else:
            diff = abs(i - dot_list[-1])
            if diff > tol :
                dot_list.append(i)
            #not similar then add
    return dot_list



def cases_overlaping(x_input, h_input, tau_min=-10, tau_max=10, N=4000, z_stan=1e-6):
    x_func = real_change(x_input)
    h_func = real_change(h_input)
    grid = np.linspace(tau_min, tau_max, N)

    support_x = support_estimation(x_func, grid, z_stan=z_stan)
    support_h = support_estimation(h_func, grid, z_stan=z_stan)

    if support_x is None :
        raise ValueError ("No support exist in x(t)...")
    elif support_h is None :
        raise ValueError ("No support exist in h(t)...")
    else :
        pass

    ax, bx = support_x
    ah, bh = support_h

    Wx = bx - ax #upper - lower
    Wh = bh - ah

    #possible boundary points
    possible_bp = pick_one([ax+ah, ax+bh, bx+ah, bx+bh], tol=1e-12)

    boundary_cases = []
    labels = []

    #before first boundary point --> very left region
    boundary_cases.append(possible_bp[0] - 1)
    labels.append("No overlap (left side)")

    #selecting mid as a representative of each boundary range
    def mid(a,b):
        return (a+b) * 0.5 
    
    if len(possible_bp) == 4: #normal case
        boundary_cases.append(mid(possible_bp[0], possible_bp[1]))
        labels.append("Head overlap")

        boundary_cases.append(mid(possible_bp[1], possible_bp[2]))
        labels.append("Full overlap")

        boundary_cases.append(mid(possible_bp[2], possible_bp[3]))
        labels.append("Tail overlap")

    elif len(possible_bp) == 3: #x(t) h(t) width same
        boundary_cases.append(mid(possible_bp[0], possible_bp[1]))
        labels.append("Head overlap")

        boundary_cases.append(mid(possible_bp[1], possible_bp[2]))
        labels.append("Tail  overlap")
    
    #after lasat boundary point --> very right region
    boundary_cases.append(possible_bp[-1] + 1)
    labels.append("No overlap (right side)")

    #prepare integral, slicing 
    tau=np.linspace(tau_min, tau_max, N)
    x_tau = x_func(tau)
    n = len(boundary_cases)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 2.4*n),sharex=True) #if fig over 2.6, crash with x-axis

    #convolution
    for i in range(len(boundary_cases)):
        t = boundary_cases[i] #time shifting t
        lab = labels[i]
        h_shift = h_func(t-tau) #h(t-τ)
        overlap = (np.abs(x_tau) > z_stan) & (np.abs(h_shift) > z_stan) #have support --> can convolute
        production = x_tau * h_shift

        axL = axes[i][0] #left side graph
        axR = axes[i][1] #right

        axL.plot(tau, x_tau, label="x(τ)")
        axL.plot(tau, h_shift, label="h(t-τ), t=" + format(t, ".3g"))

        #left graph
        ymin, ymax= axL.get_ylim() #value of y max & min, lib func
        axL.fill_between(tau, ymin, ymax, where=overlap, alpha=0.2, interpolate=True) #coloring, lib func
        axL.set_title("case " + str(i+1) + ": " +str(lab), fontsize=7)
        axL.set_ylabel("amplitude")
        axL.grid(True) #add patern shape background
        axL.legend(loc = "upper right") #position of letter

        #right graph
        axR.plot(tau, production, label = "s")
        axR.fill_between(tau, 0, production, where=overlap, alpha=0.2, interpolate=True)
        axR.set_title("Integrated on overlap", fontsize=7)
        axR.grid(True)
        axR.legend(loc = "upper right")

    for axrow in axes:
        axrow[0].tick_params(axis="x", labelbottom=True)
        axrow[1].tick_params(axis="x", labelbottom=True)

    axes[-1][0].set_xlabel("τ")
    axes[-1][1].set_xlabel("τ")  

    total_case = []
    for k in boundary_cases:
        total_case.append(format(k, ".3g"))

    plt.tight_layout(rect=[0,0,1,0.98])
    plt.show()

#run
if __name__ == "__main__":
    print("\n ---Convolution Overlap Visualizer---")
    print("Example Input Suggestion, *try this*")
    print("x(t) : u(t)")
    print("h(t) : u(t)-u(t-2)")
    print("tau_min : -1")
    print("tau_max : 5")
    print("N(slicing) : 2500\n")

    x_input = input("x(t): ")
    h_input = input("h(t): ")
    tau_min = float(input("tau_min: "))
    tau_max = float(input("tau_max: "))
    N=int(input("N(slicing): "))
    
    cases_overlaping(x_input, h_input, tau_min, tau_max, N)
    





