const fillDivClasseses = ["widget-subarea", "lm-Widget", "leaflet-container"];

function requiresFill(el) {
  if (el.tagName === "DIV") {
    return fillDivClasseses.some((cls) => {
      return el.classList.contains(cls);
    });
  }
  return false;
}

function ensureWidgetFills(el) {
  if (!el.classList.contains("html-fill-item")) {
    el.classList.add("html-fill-item");
  }

  if (!el.classList.contains("html-fill-container")) {
    el.classList.add("html-fill-container");
  }
}

function ensureWidgetsFill() {
  // Find any jupyter widget containers and keep an eye on them
  const widgetNodes = document.querySelectorAll(".widget-subarea");
  for (const widgetEl of widgetNodes) {
    ensureWidgetFills(widgetEl);
  }
}

function manageOverflow() {
  // Don't let vega cells scroll internally
  const cellOutputs = document.querySelectorAll(".cell-output-display div");
  for (const cellOutput of cellOutputs) {
    if (cellOutput.id.startsWith("altair-viz-")) {
      cellOutput.parentElement.classList.add("no-overflow-x");
    }
  }
}

function refreshStickyHeaders() {
  // Deal with markdown tables
  const markdownTables = document.querySelectorAll(".card-body > table");
  for (const markdownTable of markdownTables) {
    const scrollableArea = markdownTable.parentElement;
    stickyThead.apply([markdownTable], { scrollableArea: scrollableArea });
  }

  // Deal with iTables tables
  const cellOutputNodes = document.querySelectorAll(".card-body .cell-output");
  for (const cellOutputNode of cellOutputNodes) {
    const iTable = cellOutputNode.querySelector(".itables table");
    if (iTable) {
      stickyThead.apply([iTable], { scrollableArea: cellOutputNode });
    }
  }
}

function updatePageFlow(scrolling) {
  const dashboardContainerEl = document.querySelector(
    ".quarto-dashboard-content"
  );
  const tabContainerEl = document.querySelector(
    ".quarto-dashboard-content > .tab-content"
  );

  // update the container and body classes
  if (scrolling) {
    dashboardContainerEl.classList.add("dashboard-scrolling");
    document.body.classList.remove("dashboard-fill");
    dashboardContainerEl.classList.remove("bslib-page-fill");

    if (tabContainerEl !== null && tabContainerEl.gridTemplateRows !== null) {
      tabContainerEl.style.gridTemplateRows = "minmax(3em, max-content)";
    }
  } else {
    dashboardContainerEl.classList.remove("dashboard-scrolling");
    document.body.classList.add("dashboard-fill");
    dashboardContainerEl.classList.add("bslib-page-fill");

    if (tabContainerEl !== null && tabContainerEl.gridTemplateRows !== null) {
      tabContainerEl.style.gridTemplateRows = "minmax(3em, 1fr)";
    }
  }
}
window.document.documentElement.classList.add("hidden");
window.document.addEventListener("DOMContentLoaded", function (_event) {
  ensureWidgetsFill();

  manageOverflow();
  refreshStickyHeaders();

  // Fixup any sharing links that require urls
  // Append url to any sharing urls
  const sharingLinks = window.document.querySelectorAll(
    "a.quarto-dashboard-link"
  );
  for (let i = 0; i < sharingLinks.length; i++) {
    const sharingLink = sharingLinks[i];
    const href = sharingLink.getAttribute("href");
    if (href) {
      sharingLink.setAttribute(
        "href",
        href.replace("|url|", window.location.href)
      );
    }
  }

  // Try to process the hash and activate a tab
  const hash = window.decodeURIComponent(window.location.hash);
  if (hash.length > 0) {
    QuartoDashboardUtils.showPage(hash, () => {
      window.document.documentElement.classList.remove("hidden");
    });
  } else {
    window.document.documentElement.classList.remove("hidden");
  }

  // navigate to a tab when the history changes
  window.addEventListener("popstate", function (e) {
    const hash = window.decodeURIComponent(window.location.hash);
    QuartoDashboardUtils.showPage(hash);
  });

  // Hook tabs and use that to update history / active tabs
  const navItems = document.querySelectorAll(".navbar .nav-item .nav-link");
  for (const navItem of navItems) {
    const linkHref = navItem.getAttribute("href");
    navItem.addEventListener("click", () => {
      const baseUrl = QuartoDashboardUtils.urlWithoutHash(window.location.href);
      const hash = QuartoDashboardUtils.urlHash(linkHref);
      const href = baseUrl + hash;
      QuartoDashboardUtils.setLocation(href);

      const scrolling = navItem.getAttribute("data-scrolling");
      if (scrolling !== null) {
        updatePageFlow(scrolling.toLowerCase() === "true");
      }

      return false;
    });
  }

  // Hook links in the body so users can link to pages
  const linkEls = document.querySelectorAll(
    ".quarto-dashboard-content a:not(.nav-link)"
  );
  for (const linkEl of linkEls) {
    const linkHref = linkEl.getAttribute("href");
    linkEl.addEventListener("click", () => {
      QuartoDashboardUtils.showPage(linkHref);
      return false;
    });
  }
  const sidebar = window.document.querySelector(
    ".quarto-dashboard-content .bslib-sidebar-layout"
  );
  let prevWidth = window.document.body.clientWidth;
  const sidebarCollapseClass = "sidebar-collapsed";
  if (sidebar) {
    const resizeObserver = new ResizeObserver(
      throttle(function () {
        const clientWidth = window.document.body.clientWidth;
        if (prevWidth !== clientWidth) {
          if (clientWidth <= 576) {
            // Hide the sidebar
            if (!sidebar.classList.contains(sidebarCollapseClass)) {
              sidebar.classList.add(sidebarCollapseClass);
            }
          } else {
            // Show the sidebar
            if (sidebar.classList.contains(sidebarCollapseClass)) {
              sidebar.classList.remove(sidebarCollapseClass);
            }
          }
          prevWidth = clientWidth;
        }
      }, 2)
    );
    resizeObserver.observe(window.document.body);
  }

  const observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
      mutation.addedNodes.forEach(function (addedNode) {
        if (requiresFill(addedNode)) {
          ensureWidgetFills(addedNode);
        }
      });
    });
  });
  observer.observe(document.body, { childList: true, subtree: true });
});

// utils
window.QuartoDashboardUtils = {
  setLocation: function (href) {
    if (history && history.pushState) {
      history.pushState({}, null, href);
      // post "hashchange" for tools looking for it
      if (window.parent?.postMessage) {
        window.parent.postMessage(
          {
            type: "hashchange",
            href: window.location.href,
          },
          "*"
        );
      }
    } else {
      window.location.replace(href);
    }
    setTimeout(function () {
      window.scrollTo(0, 0);
    }, 10);
  },
  isPage: function (hash) {
    const tabPaneEl = document.querySelector(`.dashboard-page.tab-pane${hash}`);
    return tabPaneEl !== null;
  },
  showPage: function (hash, fnCallback) {
    // If the hash is empty, just select the first tab and activate that
    if (hash === "") {
      const firstTabPaneEl = document.querySelector(".dashboard-page.tab-pane");
      if (firstTabPaneEl !== null) {
        hash = `#${firstTabPaneEl.id}`;
      }
    }

    // Find the tab and activate it
    const tabNodes = document.querySelectorAll(".navbar .nav-item .nav-link");
    for (const tabEl of tabNodes) {
      const target = tabEl.getAttribute("data-bs-target");
      if (target === hash) {
        const scrolling = tabEl.getAttribute("data-scrolling");
        if (scrolling !== null) {
          updatePageFlow(scrolling.toLowerCase() === "true");
        }

        tabEl.classList.add("active");
      } else {
        tabEl.classList.remove("active");
      }
    }

    // Find the tabpanes and activate the hash tab
    const tabPaneNodes = document.querySelectorAll(".dashboard-page.tab-pane");
    for (const tabPaneEl of tabPaneNodes) {
      if (`#${tabPaneEl.id}` === hash) {
        tabPaneEl.classList.add("active");
      } else {
        tabPaneEl.classList.remove("active");
      }
    }

    if (fnCallback) {
      fnCallback();
    }
  },
  showLinkedValue: function (href) {
    // check for a page link
    if (this.isPage(href)) {
      this.showPage(href);
    } else {
      window.open(href);
    }
  },
  urlWithoutHash: function (url) {
    const hashLoc = url.indexOf("#");
    if (hashLoc != -1) return url.substring(0, hashLoc);
    else return url;
  },
  urlHash: function (url) {
    const hashLoc = url.indexOf("#");
    if (hashLoc != -1) return url.substring(hashLoc);
    else return "";
  },
};

function throttle(func, wait) {
  let waiting = false;
  return function () {
    if (!waiting) {
      func.apply(this, arguments);
      waiting = true;
      setTimeout(function () {
        waiting = false;
      }, wait);
    }
  };
}
