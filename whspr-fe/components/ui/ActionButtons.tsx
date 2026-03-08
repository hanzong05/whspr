type Action = "view" | "edit" | "delete";

type Props = {
  onView?: () => void;
  onEdit?: () => void;
  onDelete?: () => void;
  actions?: Action[];
};

const config: Record<
  Action,
  {
    label: string;
    colorClass: string;
    icon: React.ReactNode;
  }
> = {
  view: {
    label: "View",
    colorClass: "hover:text-indigo-500 hover:bg-indigo-50",
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
        />
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
        />
      </svg>
    ),
  },
  edit: {
    label: "Edit",
    colorClass: "hover:text-blue-500 hover:bg-blue-50",
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
        />
      </svg>
    ),
  },
  delete: {
    label: "Delete",
    colorClass: "hover:text-red-500 hover:bg-red-50",
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
        />
      </svg>
    ),
  },
};

export default function ActionButtons({
  onView,
  onEdit,
  onDelete,
  actions = ["view", "edit", "delete"],
}: Props) {
  const handlers: Record<Action, (() => void) | undefined> = {
    view: onView,
    edit: onEdit,
    delete: onDelete,
  };

  return (
    <div className="flex items-center gap-1">
      {actions.map((action) => {
        const handler = handlers[action];
        const { label, colorClass, icon } = config[action];
        return (
          <button
            key={action}
            onClick={handler}
            title={label}
            disabled={!handler}
            className={`p-1.5 text-gray-400 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed ${colorClass}`}
          >
            {icon}
          </button>
        );
      })}
    </div>
  );
}
