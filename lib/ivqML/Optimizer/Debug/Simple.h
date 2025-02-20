// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Degug__Adam__h__
#define __ivqML__Optimizer__Degug__Adam__h__

namespace ivqML
{
  namespace Optimizer
  {
    namespace Debug
    {
      /**
       */
      template< class _TModel >
      class Simple
      {
      public:
        using TModel   = _TModel;
        using Self     = Simple;
        using TReal    = typename TModel::TReal;
        using TNatural = typename TModel::TNatural;
        using TMatrix  = typename TModel::TMatrix;
        using TColumn  = typename TModel::TColumn;
        using TRow     = typename TModel::TRow;

      public:
        Simple( const TModel& m, std::ostream& o, const TNatural& e )
          : m_Model( &m ),
            m_Out( &o ),
            m_Epochs( e )
          {
            this->m_Epsilon = std::numeric_limits< TReal >::epsilon( );
          }
        virtual ~Simple( )
          {
          }

        virtual bool operator()(
          const TNatural& t,
          const TReal& nG,
          const TReal& J_tr, const TReal& J_te
          )
          {
            *( this->m_Out )
              << t << " " << nG << " " << J_tr << " " << J_te
              << std::endl;
            return( !( t < this->m_Epochs && this->m_Epsilon <= nG  ) );
          }

      protected:
        const TModel* m_Model;
        std::ostream* m_Out;
        TNatural m_Epochs;
        TReal m_Epsilon;
      };
    } // end namespace
  } // end namespace
} // end namespace

#endif // __ivqML__Optimizer__Degug__Simple__h__

// eof - Simple.h
