// OptiCrop Main Interactive Script

document.addEventListener('DOMContentLoaded', function () {
    // Synchronize number inputs and range sliders
    const syncPairs = [
        { range: 'n_range', input: 'n_input' },
        { range: 'p_range', input: 'p_input' },
        { range: 'k_range', input: 'k_input' },
        { range: 'temp_range', input: 'temp_input' },
        { range: 'humidity_range', input: 'humidity_input' },
        { range: 'ph_range', input: 'ph_input' },
        { range: 'rainfall_range', input: 'rainfall_input' }
    ];

    syncPairs.forEach(pair => {
        const rangeEl = document.getElementById(pair.range);
        const inputEl = document.getElementById(pair.input);

        if (rangeEl && inputEl) {
            rangeEl.addEventListener('input', function () {
                inputEl.value = this.value;
            });
            inputEl.addEventListener('input', function () {
                rangeEl.value = this.value;
            });
        }
    });

    // Preset values for quick testing
    const presets = {
        'rice': { n: 90, p: 42, k: 43, temp: 20.88, humidity: 82.0, ph: 6.5, rainfall: 202.9 },
        'cotton': { n: 118, p: 47, k: 19, temp: 23.96, humidity: 79.8, ph: 6.9, rainfall: 80.3 },
        'coffee': { n: 101, p: 29, k: 30, temp: 25.54, humidity: 57.7, ph: 6.8, rainfall: 158.0 },
        'apple': { n: 21, p: 134, k: 199, temp: 22.3, humidity: 92.3, ph: 5.9, rainfall: 112.5 },
        'chickpea': { n: 40, p: 67, k: 79, temp: 18.9, humidity: 16.9, ph: 7.3, rainfall: 80.1 }
    };

    window.applyPreset = function (cropName) {
        const data = presets[cropName];
        if (!data) return;

        setParam('n', data.n);
        setParam('p', data.p);
        setParam('k', data.k);
        setParam('temp', data.temp);
        setParam('humidity', data.humidity);
        setParam('ph', data.ph);
        setParam('rainfall', data.rainfall);
    };

    function setParam(name, val) {
        const r = document.getElementById(`${name}_range`);
        const i = document.getElementById(`${name}_input`);
        if (r) r.value = val;
        if (i) i.value = val;
    }
});
