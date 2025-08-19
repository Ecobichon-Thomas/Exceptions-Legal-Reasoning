document.addEventListener("DOMContentLoaded", function () {
    let scrollPosition = sessionStorage.getItem("scrollPosition");
    if (scrollPosition) {
        window.scrollTo(0, scrollPosition);
        sessionStorage.removeItem("scrollPosition");
    }

    window.addEventListener("beforeunload", function() {
        sessionStorage.setItem("scrollPosition", window.scrollY);
    });

    const autoForm = document.getElementById("auto-submit-form");
    if (autoForm) autoForm.submit();

    const wSelect = document.getElementById("w_choice");
    const wUpload = document.getElementById("w_upload_fields");
    if (wSelect && wUpload) {
        wSelect.addEventListener("change", () => {
            wUpload.style.display = wSelect.value === "__upload__" ? "flex" : "none";
        });
    }

    const rbSelect = document.getElementById("rb_choice");
    const rbUpload = document.getElementById("rb_upload_fields");
    if (rbSelect && rbUpload) {
        rbSelect.addEventListener("change", () => {
            rbUpload.style.display = rbSelect.value === "__upload__" ? "flex" : "none";
        });
    }

    const uploadedW = document.getElementById("uploaded_w");
    if (uploadedW) {
        uploadedW.addEventListener("change", function () {
            document.getElementById("w_file_name").textContent =
                this.files.length ? this.files[0].name : "";
        });
    }

    const uploadedRB = document.getElementById("uploaded_rb");
    if (uploadedRB) {
        uploadedRB.addEventListener("change", function () {
            document.getElementById("rb_file_name").textContent =
                this.files.length ? this.files[0].name : "";
        });
    }

    const radioButtons = document.querySelectorAll('input[name="selection"]');
    const seuilContainer = document.getElementById("seuil-container");
    const seuilInput = document.getElementById("seuil");

    function updateSeuilVisibility() {
        const selected = document.querySelector('input[name="selection"]:checked');
        if (selected && (selected.value === "Seuil" || selected.value === "Seuil Minimal")) {
            seuilContainer.style.display = "block";
            seuilInput.disabled = false;
        } else {
            seuilContainer.style.display = "none";
            seuilInput.disabled = true;
        }
    }

    if (radioButtons.length > 0) {
        radioButtons.forEach(btn => btn.addEventListener("change", updateSeuilVisibility));
        updateSeuilVisibility();
    }

    document.querySelectorAll(".section__toggle").forEach(toggle => {
        toggle.addEventListener("click", () => {
            const section = toggle.closest(".section");
            section.classList.toggle("is-collapsed");
            toggle.textContent = section.classList.contains("is-collapsed") ? "+" : "–";
        });
    });
});
