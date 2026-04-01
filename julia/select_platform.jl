using OpenCL

platform = cl.platforms()[1]  # seleziona la seconda piattaforma
device = cl.devices(platform)[1]  # ad esempio il primo device su quella piattaforma

ctx = cl.Context(device)
queue = cl.CmdQueue(ctx)

@info "Piattaforma selezionata: $(OpenCL.name(platform))"
@info "Dispositivo selezionato: $(OpenCL.name(device))"
