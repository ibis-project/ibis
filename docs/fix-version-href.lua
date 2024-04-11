function Header(el)
  -- Walk the pandoc AST and find all links in Header elements
  -- Pluck out the version string and set it if it hasn't been set already
  local version = nil;

  el = el:walk({
    Link = function(link)
      return link:walk({
        Str = function(str)
          if version == nil then
            version = str.text
          end
          return str.text
        end,
      })
    end,
  })

  el.attr.identifier = version

  return el
end
