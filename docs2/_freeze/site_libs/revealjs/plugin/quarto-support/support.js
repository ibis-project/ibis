// catch all plugin for various quarto features
window.QuartoSupport = function () {
  function isPrintView() {
    return /print-pdf/gi.test(window.location.search);
  }

  // implement controlsAudo
  function controlsAuto(deck) {
    const config = deck.getConfig();
    if (config.controlsAuto === true) {
      const iframe = window.location !== window.parent.location;
      const localhost =
        window.location.hostname === "localhost" ||
        window.location.hostname === "127.0.0.1";
      deck.configure({
        controls:
          (iframe && !localhost) ||
          (deck.hasVerticalSlides() && config.navigationMode !== "linear"),
      });
    }
  }

  // helper to provide event handlers for all links in a container
  function handleLinkClickEvents(deck, container) {
    Array.from(container.querySelectorAll("a")).forEach((el) => {
      const url = el.getAttribute("href");
      if (/^(http|www)/gi.test(url)) {
        el.addEventListener(
          "click",
          (ev) => {
            const fullscreen = !!window.document.fullscreen;
            const dataPreviewLink = el.getAttribute("data-preview-link");

            // if there is a local specifcation then use that
            if (dataPreviewLink) {
              if (
                dataPreviewLink === "true" ||
                (dataPreviewLink === "auto" && fullscreen)
              ) {
                ev.preventDefault();
                deck.showPreview(url);
                return false;
              }
            } else {
              const previewLinks = !!deck.getConfig().previewLinks;
              const previewLinksAuto =
                deck.getConfig().previewLinksAuto === true;
              if (previewLinks == true || (previewLinksAuto && fullscreen)) {
                ev.preventDefault();
                deck.showPreview(url);
                return false;
              }
            }

            // if the deck is in an iframe we want to open it externally
            // (don't do this when in vscode though as it has its own
            // handler for opening links externally that will be play)
            const iframe = window.location !== window.parent.location;
            if (
              iframe &&
              !window.location.search.includes("quartoPreviewReqId=")
            ) {
              ev.preventDefault();
              ev.stopImmediatePropagation();
              window.open(url, "_blank");
              return false;
            }

            // if the user has set data-preview-link to "auto" we need to handle the event
            // (because reveal will interpret "auto" as true)
            if (dataPreviewLink === "auto") {
              ev.preventDefault();
              ev.stopImmediatePropagation();
              const target =
                el.getAttribute("target") ||
                (ev.ctrlKey || ev.metaKey ? "_blank" : "");
              if (target) {
                window.open(url, target);
              } else {
                window.location.href = url;
              }
              return false;
            }
          },
          false
        );
      }
    });
  }

  // implement previewLinksAuto
  function previewLinksAuto(deck) {
    handleLinkClickEvents(deck, deck.getRevealElement());
  }

  // apply styles
  function applyGlobalStyles(deck) {
    if (deck.getConfig()["smaller"] === true) {
      const revealParent = deck.getRevealElement();
      revealParent.classList.add("smaller");
    }
  }

  // add logo image
  function addLogoImage(deck) {
    const revealParent = deck.getRevealElement();
    const logoImg = document.querySelector(".slide-logo");
    if (logoImg) {
      revealParent.appendChild(logoImg);
      revealParent.classList.add("has-logo");
    }
  }

  // add footer text
  function addFooter(deck) {
    const revealParent = deck.getRevealElement();
    const defaultFooterDiv = document.querySelector(".footer-default");
    if (defaultFooterDiv) {
      revealParent.appendChild(defaultFooterDiv);
      handleLinkClickEvents(deck, defaultFooterDiv);
      if (!isPrintView()) {
        deck.on("slidechanged", function (ev) {
          const prevSlideFooter = document.querySelector(
            ".reveal > .footer:not(.footer-default)"
          );
          if (prevSlideFooter) {
            prevSlideFooter.remove();
          }
          const currentSlideFooter = ev.currentSlide.querySelector(".footer");
          if (currentSlideFooter) {
            defaultFooterDiv.style.display = "none";
            const slideFooter = currentSlideFooter.cloneNode(true);
            handleLinkClickEvents(deck, slideFooter);
            deck.getRevealElement().appendChild(slideFooter);
          } else {
            defaultFooterDiv.style.display = "block";
          }
        });
      }
    }
  }

  // add chalkboard buttons
  function addChalkboardButtons(deck) {
    const chalkboard = deck.getPlugin("RevealChalkboard");
    if (chalkboard && !isPrintView()) {
      const revealParent = deck.getRevealElement();
      const chalkboardDiv = document.createElement("div");
      chalkboardDiv.classList.add("slide-chalkboard-buttons");
      if (document.querySelector(".slide-menu-button")) {
        chalkboardDiv.classList.add("slide-menu-offset");
      }
      // add buttons
      const buttons = [
        {
          icon: "easel2",
          title: "Toggle Chalkboard (b)",
          onclick: chalkboard.toggleChalkboard,
        },
        {
          icon: "brush",
          title: "Toggle Notes Canvas (c)",
          onclick: chalkboard.toggleNotesCanvas,
        },
      ];
      buttons.forEach(function (button) {
        const span = document.createElement("span");
        span.title = button.title;
        const icon = document.createElement("i");
        icon.classList.add("fas");
        icon.classList.add("fa-" + button.icon);
        span.appendChild(icon);
        span.onclick = function (event) {
          event.preventDefault();
          button.onclick();
        };
        chalkboardDiv.appendChild(span);
      });
      revealParent.appendChild(chalkboardDiv);
      const config = deck.getConfig();
      if (!config.chalkboard.buttons) {
        chalkboardDiv.classList.add("hidden");
      }

      // show and hide chalkboard buttons on slidechange
      deck.on("slidechanged", function (ev) {
        const config = deck.getConfig();
        let buttons = !!config.chalkboard.buttons;
        const slideButtons = ev.currentSlide.getAttribute(
          "data-chalkboard-buttons"
        );
        if (slideButtons) {
          if (slideButtons === "true" || slideButtons === "1") {
            buttons = true;
          } else if (slideButtons === "false" || slideButtons === "0") {
            buttons = false;
          }
        }
        if (buttons) {
          chalkboardDiv.classList.remove("hidden");
        } else {
          chalkboardDiv.classList.add("hidden");
        }
      });
    }
  }

  function handleTabbyClicks() {
    const tabs = document.querySelectorAll(".panel-tabset-tabby > li > a");
    for (let i = 0; i < tabs.length; i++) {
      const tab = tabs[i];
      tab.onclick = function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        return false;
      };
    }
  }

  function fixupForPrint(deck) {
    if (isPrintView()) {
      const slides = deck.getSlides();
      slides.forEach(function (slide) {
        slide.removeAttribute("data-auto-animate");
      });
      window.document.querySelectorAll(".hljs").forEach(function (el) {
        el.classList.remove("hljs");
      });
      window.document.querySelectorAll(".hljs-ln-code").forEach(function (el) {
        el.classList.remove("hljs-ln-code");
      });
    }
  }

  function handleSlideChanges(deck) {
    // dispatch for htmlwidgets
    const fireSlideEnter = () => {
      const event = window.document.createEvent("Event");
      event.initEvent("slideenter", true, true);
      window.document.dispatchEvent(event);
    };

    const fireSlideChanged = (previousSlide, currentSlide) => {
      fireSlideEnter();

      // dispatch for shiny
      if (window.jQuery) {
        if (previousSlide) {
          window.jQuery(previousSlide).trigger("hidden");
        }
        if (currentSlide) {
          window.jQuery(currentSlide).trigger("shown");
        }
      }
    };

    // fire slideEnter for tabby tab activations (for htmlwidget resize behavior)
    document.addEventListener("tabby", fireSlideEnter, false);

    deck.on("slidechanged", function (event) {
      fireSlideChanged(event.previousSlide, event.currentSlide);
    });
  }

  function workaroundMermaidDistance(deck) {
    if (window.document.querySelector("pre.mermaid-js")) {
      const slideCount = deck.getTotalSlides();
      deck.configure({
        mobileViewDistance: slideCount,
        viewDistance: slideCount,
      });
    }
  }

  return {
    id: "quarto-support",
    init: function (deck) {
      controlsAuto(deck);
      previewLinksAuto(deck);
      fixupForPrint(deck);
      applyGlobalStyles(deck);
      addLogoImage(deck);
      addFooter(deck);
      addChalkboardButtons(deck);
      handleTabbyClicks();
      handleSlideChanges(deck);
      workaroundMermaidDistance(deck);
    },
  };
};
