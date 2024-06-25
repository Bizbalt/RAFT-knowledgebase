const url = window.location.origin + "/"


// building the dropdown menus
async function options_innit() {
    const response = await fetch(url + 'get_dropdown_options')
    const dropdown_options_dic = await response.json()

    const monomerSel = document.getElementById("monomer");
    const raft_agentSel = document.getElementById("raft_agent");
    const solventSel = document.getElementById("solvent");

    const dropdownSelectors = {"monomer": monomerSel, "raft_agent": raft_agentSel, "solvent": solventSel}
    const keys = ["monomer", "raft_agent", "solvent"]

    for (const dropdown of keys){
        const dropdownSel = dropdownSelectors[dropdown]
        for (const option of dropdown_options_dic[dropdown]){
            dropdownSel.options[dropdownSel.options.length] = new Option(option, option);
    }
    }
}

window.onload = function() {
  options_innit().then(r => console.log("Dropdowns initialized"))
}


// function to send a plot request to the server
async function plot_exp() {
    let experiment_list = document.getElementById("search").value;
    // create a list of the experiments numbers as strings
    let experiments_list = experiment_list.split(",").map(x => x.trim()).join(",")
    let plot_mn = null
    let plot_mw = null
    let fit_curves = null
    const response = await fetch(url + `plot_exp`, {
        method: "POST",
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ exp_nr: experiments_list, plot_mn: plot_mn, plot_mw: plot_mw, fit_curves: fit_curves })
        })
    const results_div = document.getElementById("results_div");
    results_div.inner = response.text()

}
